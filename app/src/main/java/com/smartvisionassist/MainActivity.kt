package com.smartvisionassist

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.accessibility.AccessibilityEvent
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.smartvisionassist.camera.CameraAnalyzer
import com.smartvisionassist.databinding.ActivityMainBinding
import com.smartvisionassist.dev.DevSettings
import com.smartvisionassist.ml.DepthProcessor
import com.smartvisionassist.ml.HazardProcessor
import com.smartvisionassist.ml.ModelManager
import com.smartvisionassist.ml.YoloProcessor
import com.smartvisionassist.navigation.NavigationEngine
import com.smartvisionassist.navigation.SceneMemory
import com.smartvisionassist.speech.TTSManager
import com.smartvisionassist.tracking.ObjectTracker
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var modelManager: ModelManager
    private lateinit var ttsManager: TTSManager

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val devSettings = DevSettings()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        modelManager = ModelManager(this)
        modelManager.logRuntimeMode()
        ttsManager = TTSManager(this)

        binding.devToggle.setOnCheckedChangeListener { _, checked ->
            devSettings.enabled = checked
            binding.fpsText.alpha = if (checked) 1f else 0f
        }
        binding.fpsText.alpha = 0f

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val cameraProvider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val yoloLabels = listOf("person", "vehicle", "animal", "chair", "object")
            val hazardLabels = listOf("pothole", "pole", "staircase", "ramp", "wall")

            val analyzer = CameraAnalyzer(
                yoloProcessor = YoloProcessor(modelManager.yoloInterpreter, yoloLabels),
                hazardProcessor = HazardProcessor(modelManager.hazardInterpreter, hazardLabels),
                depthProcessor = DepthProcessor(modelManager.depthInterpreter),
                tracker = ObjectTracker(),
                navigationEngine = NavigationEngine(),
                sceneMemory = SceneMemory(),
                inferenceExecutor = inferenceExecutor,
                devSettings = devSettings,
                onFrameResult = { command, fps, _ ->
                    runOnUiThread {
                        binding.commandText.text = command
                        binding.commandText.contentDescription = command
                        binding.commandText.sendAccessibilityEvent(AccessibilityEvent.TYPE_VIEW_FOCUSED)
                        binding.fpsText.text = "FPS: ${"%.1f".format(fps)}"
                    }
                },
                onSpeak = { command -> ttsManager.speakLatest(command) }
            )

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor, analyzer) }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        inferenceExecutor.shutdown()
        ttsManager.shutdown()
        modelManager.close()
    }
}
