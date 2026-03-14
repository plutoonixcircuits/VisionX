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
            for (i in 0 until output[0][0].size step 30) {
                val score = output[0][4][i]
                if (score < 0.3f) continue
                val cx = output[0][0][i] * imageW
                val cy = output[0][1][i] * imageH
                val bw = output[0][2][i] * imageW
                val bh = output[0][3][i] * imageH
                val classIdx = 5 + labels.indices.maxByOrNull { idx -> output[0][5 + idx][i] }!!.toInt()
                results += DetectedObject(
                    label = labels.getOrElse(classIdx - 5) { "hazard" },
                    score = score,
                    left = cx - bw / 2,
                    top = cy - bh / 2,
                    right = cx + bw / 2,
                    bottom = cy + bh / 2
                )
            }
            results
        }
}
