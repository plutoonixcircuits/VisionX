package com.smartvisionassist.camera

import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.smartvisionassist.dev.DevSettings
import com.smartvisionassist.ml.DepthProcessor
import com.smartvisionassist.ml.DetectedObject
import com.smartvisionassist.ml.HazardProcessor
import com.smartvisionassist.ml.ModelManager
import com.smartvisionassist.ml.YoloProcessor
import com.smartvisionassist.navigation.NavigationEngine
import com.smartvisionassist.navigation.SceneMemory
import com.smartvisionassist.navigation.SpatialGrid
import com.smartvisionassist.tracking.ObjectTracker
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.system.measureTimeMillis

class CameraAnalyzer(
    private val yoloProcessor: YoloProcessor,
    private val hazardProcessor: HazardProcessor,
    private val depthProcessor: DepthProcessor,
    private val tracker: ObjectTracker,
    private val navigationEngine: NavigationEngine,
    private val sceneMemory: SceneMemory,
    private val inferenceExecutor: ExecutorService,
    private val devSettings: DevSettings,
    private val onFrameResult: (command: String, fps: Float, objects: List<DetectedObject>) -> Unit,
    private val onSpeak: (String) -> Unit
) : ImageAnalysis.Analyzer {

    private val inFlight = AtomicBoolean(false)
    private var frameIndex = 0
    private var frameTimes = ArrayDeque<Long>()

    private val yoloInput: ByteBuffer = ByteBuffer
        .allocateDirect(1 * ModelManager.MODEL_INPUT_SIZE * ModelManager.MODEL_INPUT_SIZE * 3 * 4)
        .order(ByteOrder.nativeOrder())
    private val depthInput: ByteBuffer = ByteBuffer
        .allocateDirect(1 * ModelManager.DEPTH_INPUT_SIZE * ModelManager.DEPTH_INPUT_SIZE * 3 * 4)
        .order(ByteOrder.nativeOrder())

    @Volatile
    private var latestDepthMap: FloatArray = FloatArray(0)

    override fun analyze(image: ImageProxy) {
        frameIndex++
        if (!inFlight.compareAndSet(false, true)) {
            image.close()
            return
        }

        val width = image.width
        val height = image.height
        val yPlane = image.planes[0].buffer

        fillInputBufferFromY(yPlane, yoloInput, ModelManager.MODEL_INPUT_SIZE)
        fillInputBufferFromY(yPlane, depthInput, ModelManager.DEPTH_INPUT_SIZE)
        image.close()

        inferenceExecutor.execute {
            val elapsed = measureTimeMillis {
                val yoloDetections = if (frameIndex % 2 == 0) yoloProcessor.run(yoloInput, width, height) else emptyList()
                val hazardDetections = if (frameIndex % 3 == 0) hazardProcessor.run(yoloInput, width, height) else emptyList()
                if (frameIndex % 4 == 0) {
                    latestDepthMap = depthProcessor.run(depthInput)
                }

                val merged = (yoloDetections + hazardDetections).map { d ->
                    val zone = SpatialGrid.zoneFor(d.cx, d.cy, width, height)
                    val distance = depthProcessor.estimateDistance(
                        latestDepthMap,
                        d.cx,
                        d.cy,
                        width,
                        height,
                        devSettings.calibrationFactor
                    )
                    d.copy(distanceMeters = distance, gridZone = zone)
                }

                val tracked = tracker.update(merged)
                if (devSettings.enabled && devSettings.logDetections) {
                    tracked.forEach {
                        Log.d("SmartVision", "ID:${it.id} Class:${it.label} Distance:${"%.2f".format(it.distanceMeters)}m Grid:${it.gridZone}")
                    }
                }

                val command = navigationEngine.buildCommand(tracked)
                val fps = updateFps()
                onFrameResult(command, fps, tracked)
                if (sceneMemory.shouldSpeak(command)) {
                    onSpeak(command)
                }
            }
            inFlight.set(false)
            if (elapsed > 70) {
                Log.d("SmartVision", "Frame pipeline lag: ${elapsed}ms")
            }
        }
    }

    private fun updateFps(): Float {
        val now = System.currentTimeMillis()
        frameTimes.addLast(now)
        while (frameTimes.isNotEmpty() && now - frameTimes.first() > 1000) {
            frameTimes.removeFirst()
        }
        return frameTimes.size.toFloat()
    }

    private fun fillInputBufferFromY(source: ByteBuffer, target: ByteBuffer, size: Int) {
        target.rewind()
        source.rewind()
        val remaining = source.remaining().coerceAtLeast(1)
        for (i in 0 until size * size) {
            val y = source.get(i % remaining).toInt() and 0xFF
            val normalized = y / 255f
            target.putFloat(normalized)
            target.putFloat(normalized)
            target.putFloat(normalized)
        }
        target.rewind()
    }
}
