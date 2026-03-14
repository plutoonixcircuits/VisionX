package com.smartvisionassist.ml

import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

class HazardProcessor(
    private val interpreter: Interpreter,
    private val labels: List<String>
) {
    private val output = Array(1) { Array(10) { FloatArray(8400) } }

    fun run(input: ByteBuffer, imageW: Int, imageH: Int): List<DetectedObject> =
        safeInfer("Hazard") {
            interpreter.run(input, output)
            val results = mutableListOf<DetectedObject>()
            val numBoxes = output[0][0].size

            for (i in 0 until numBoxes) {
                val objectness = output[0][4][i]
                if (objectness < 0.25f) continue

                var bestClass = 0
                var bestClassScore = 0f
                for (idx in labels.indices) {
                    val classScore = output[0][5 + idx][i]
                    if (classScore > bestClassScore) {
                        bestClassScore = classScore
                        bestClass = idx
                    }
                }

                val confidence = objectness * bestClassScore
                if (confidence < 0.3f) continue

                val cx = output[0][0][i] * imageW
                val cy = output[0][1][i] * imageH
                val bw = output[0][2][i] * imageW
                val bh = output[0][3][i] * imageH
                results += DetectedObject(
                    label = labels.getOrElse(bestClass) { "hazard" },
                    score = confidence,
                    left = (cx - bw / 2f).coerceAtLeast(0f),
                    top = (cy - bh / 2f).coerceAtLeast(0f),
                    right = (cx + bw / 2f).coerceAtMost(imageW.toFloat()),
                    bottom = (cy + bh / 2f).coerceAtMost(imageH.toFloat())
                )
            }
            results
        }
}
