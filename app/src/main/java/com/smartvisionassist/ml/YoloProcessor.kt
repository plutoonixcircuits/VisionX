package com.smartvisionassist.ml

import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

class YoloProcessor(
    private val interpreter: Interpreter,
    private val labels: List<String>
) {
    private val output = Array(1) { Array(84) { FloatArray(8400) } }

    fun run(input: ByteBuffer, imageW: Int, imageH: Int): List<DetectedObject> {
        return safeInfer("YOLO") {
            interpreter.run(input, output)
            decode(output[0], labels, imageW, imageH)
        }
    }

    private fun decode(raw: Array<FloatArray>, labels: List<String>, w: Int, h: Int): List<DetectedObject> {
        val results = mutableListOf<DetectedObject>()
        for (i in 0 until raw[0].size step 25) {
            val score = raw[4][i]
            if (score < 0.35f) continue
            val cx = raw[0][i] * w
            val cy = raw[1][i] * h
            val bw = raw[2][i] * w
            val bh = raw[3][i] * h
            val classIdx = raw.indices.drop(5).maxByOrNull { raw[it][i] } ?: 0
            val label = labels.getOrElse(classIdx - 5) { "object" }
            results += DetectedObject(
                label = label,
                score = score,
                left = (cx - bw / 2f).coerceAtLeast(0f),
                top = (cy - bh / 2f).coerceAtLeast(0f),
                right = (cx + bw / 2f).coerceAtMost(w.toFloat()),
                bottom = (cy + bh / 2f).coerceAtMost(h.toFloat())
            )
        }
        return results
    }
}

internal inline fun <T> safeInfer(tag: String, block: () -> T): T {
    return try {
        block()
    } catch (t: Throwable) {
        Log.e("SmartVision", "$tag inference failed: ${t.message}")
        when (tag) {
            "Depth" -> FloatArray(0) as T
            else -> emptyList<DetectedObject>() as T
        }
    }
}
