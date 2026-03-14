package com.smartvisionassist.navigation

object SpatialGrid {
    fun zoneFor(cx: Float, cy: Float, width: Int, height: Int): String {
        val col = when {
            cx < width / 3f -> "LEFT"
            cx < (width * 2f / 3f) -> "CENTER"
            else -> "RIGHT"
        }
        val row = when {
            cy < height / 3f -> "FAR"
            cy < (height * 2f / 3f) -> "MID"
            else -> "NEAR"
        }
        return "$row-$col"
    }
}
