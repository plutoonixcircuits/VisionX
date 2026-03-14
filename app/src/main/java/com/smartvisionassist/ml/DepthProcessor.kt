package com.smartvisionassist.ml

import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

class DepthProcessor(private val interpreter: Interpreter) {
    private val outputBuffer = Array(1) { Array(256) { FloatArray(256) } }

    fun run(input: ByteBuffer): FloatArray = safeInfer("Depth") {
        interpreter.run(input, outputBuffer)
        flatten(outputBuffer[0])
    }

    fun estimateDistance(depthMap: FloatArray, x: Float, y: Float, width: Int, height: Int, calibrationFactor: Float): Float {
        if (depthMap.isEmpty()) return -1f
        val dx = ((x / width) * 255f).toInt().coerceIn(0, 255)
        val dy = ((y / height) * 255f).toInt().coerceIn(0, 255)
        val depthValue = depthMap[(dy * 256) + dx]
        return calibrationFactor / (depthValue + 0.01f)
    }

    private fun flatten(depth: Array<FloatArray>): FloatArray {
        val flat = FloatArray(256 * 256)
        var idx = 0
        for (row in depth) {
            for (v in row) flat[idx++] = v
        }
        return flat
    }
}
