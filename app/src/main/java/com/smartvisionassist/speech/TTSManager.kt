package com.smartvisionassist.speech

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale
import java.util.concurrent.Executors

class TTSManager(context: Context) : TextToSpeech.OnInitListener {
    private val speechExecutor = Executors.newSingleThreadExecutor()
    private var tts: TextToSpeech = TextToSpeech(context.applicationContext, this)
    @Volatile
    private var ready = false

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
            ready = true
        }
    }

    fun speakLatest(text: String) {
        speechExecutor.execute {
            if (!ready) return@execute
            tts.stop()
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "nav-command")
        }
    }

    fun shutdown() {
        speechExecutor.execute {
            tts.stop()
            tts.shutdown()
        }
        speechExecutor.shutdown()
    }
}
