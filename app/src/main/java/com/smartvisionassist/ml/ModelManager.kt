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
        val model = FileUtil.loadMappedFile(context, "yolov8n_float16.tflite")
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
        Interpreter(FileUtil.loadMappedFile(context, "best_float16.tflite"), options)
    }

    val depthInterpreter: Interpreter by lazy {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setUseXNNPACK(true)
        }
        Interpreter(FileUtil.loadMappedFile(context, "midas_small.tflite"), options)
    }

    fun logRuntimeMode() {
        val mode = if (gpuEnabled) "GPU capable device (YOLO attempts GPU)" else "CPU/XNNPACK mode"
        Log.d("SmartVision", "Runtime mode: $mode")
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

    companion object {
        const val MODEL_INPUT_SIZE = 320
        const val DEPTH_INPUT_SIZE = 256
        val FLOAT_TYPE = DataType.FLOAT32
    }
}
