package com.smartvisionassist.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil

class ModelManager(private val context: Context) {

    private var gpuDelegate: GpuDelegate? = null
    private val compatibilityList = CompatibilityList()

    val gpuEnabled: Boolean by lazy {
        compatibilityList.isDelegateSupportedOnThisDevice
    }

    val yoloInterpreter: Interpreter by lazy {
        val model = FileUtil.loadMappedFile(context, resolveAssetName(YOLO_MODEL_BASENAME))
        val yoloOptions = Interpreter.Options().apply {
            setNumThreads(4)
            if (!gpuEnabled) {
                setUseXNNPACK(true)
            }
        }

        if (gpuEnabled) {
            try {
                gpuDelegate = GpuDelegate()
                yoloOptions.addDelegate(gpuDelegate)
                Log.d("SmartVision", "YOLO using GPU delegate")
            } catch (t: Throwable) {
                Log.e("SmartVision", "GPU delegate unavailable, falling back to CPU: ${t.message}")
                gpuDelegate?.close()
                gpuDelegate = null
                yoloOptions.setUseXNNPACK(true)
            }
        }

        Interpreter(model, yoloOptions)
    }

    val hazardInterpreter: Interpreter by lazy {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setUseXNNPACK(true)
        }
        Interpreter(FileUtil.loadMappedFile(context, resolveAssetName(HAZARD_MODEL_BASENAME)), options)
    }

    val depthInterpreter: Interpreter by lazy {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setUseXNNPACK(true)
        }
        Interpreter(FileUtil.loadMappedFile(context, resolveAssetName(DEPTH_MODEL_BASENAME)), options)
    }

    fun logRuntimeMode() {
        val mode = if (gpuEnabled) "GPU capable device (YOLO attempts GPU)" else "CPU/XNNPACK mode"
        Log.d("SmartVision", "Runtime mode: $mode")
        Log.d(
            "SmartVision",
            "Models: ${resolveAssetName(YOLO_MODEL_BASENAME)}, ${resolveAssetName(HAZARD_MODEL_BASENAME)}, ${resolveAssetName(DEPTH_MODEL_BASENAME)}"
        )
    }

    fun close() {
        try {
            yoloInterpreter.close()
            hazardInterpreter.close()
            depthInterpreter.close()
            gpuDelegate?.close()
        } catch (t: Throwable) {
            Log.e("SmartVision", "Safe close error: ${t.message}")
        }
    }

    private fun resolveAssetName(baseName: String): String {
        val assets = context.assets.list("")?.toSet().orEmpty()
        return when {
            assets.contains("$baseName.tflite") -> "$baseName.tflite"
            assets.contains(baseName) -> baseName
            else -> "$baseName.tflite"
        }
    }

    companion object {
        private const val YOLO_MODEL_BASENAME = "yolov8n_float16"
        private const val HAZARD_MODEL_BASENAME = "best_float16"
        private const val DEPTH_MODEL_BASENAME = "midas_small"

        const val MODEL_INPUT_SIZE = 320
        const val DEPTH_INPUT_SIZE = 256
        val FLOAT_TYPE = DataType.FLOAT32
    }
}
