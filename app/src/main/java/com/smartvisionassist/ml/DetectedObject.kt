package com.smartvisionassist.ml

data class DetectedObject(
    val id: Int = -1,
    val label: String,
    val score: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val distanceMeters: Float = -1f,
    val gridZone: String = "UNKNOWN"
) {
    val cx: Float get() = (left + right) * 0.5f
    val cy: Float get() = (top + bottom) * 0.5f
}
