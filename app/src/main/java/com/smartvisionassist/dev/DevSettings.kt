package com.smartvisionassist.dev

data class DevSettings(
    var enabled: Boolean = false,
    var calibrationFactor: Float = 2.0f,
    var showBoundingBoxes: Boolean = true,
    var showObjectId: Boolean = true,
    var showClass: Boolean = true,
    var showDistance: Boolean = true,
    var logDetections: Boolean = true
)
