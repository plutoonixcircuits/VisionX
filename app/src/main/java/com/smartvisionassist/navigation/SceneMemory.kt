package com.smartvisionassist.navigation

class SceneMemory {
    private var lastCommand: String = ""

    fun shouldSpeak(newCommand: String): Boolean {
        if (newCommand == lastCommand) return false
        lastCommand = newCommand
        return true
    }
}
