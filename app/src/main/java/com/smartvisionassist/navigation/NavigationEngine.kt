package com.smartvisionassist.navigation

import com.smartvisionassist.ml.DetectedObject

class NavigationEngine {
    fun buildCommand(objects: List<DetectedObject>): String {
        if (objects.isEmpty()) return "Path clear, move forward."

        val nearest = objects.minByOrNull { it.distanceMeters.takeIf { d -> d > 0f } ?: Float.MAX_VALUE }
            ?: return "Path clear, move forward."

        if (nearest.distanceMeters in 0f..0.5f) {
            return "${nearest.label} detected ${"%.1f".format(nearest.distanceMeters)} meters ahead. STOP! Object very close ahead."
        }

        val centerObstacle = objects.filter { it.gridZone.endsWith("CENTER") }.minByOrNull { it.distanceMeters }
        if (centerObstacle != null) {
            val leftBusy = objects.any { it.gridZone.endsWith("LEFT") && it.distanceMeters < 2f }
            val rightBusy = objects.any { it.gridZone.endsWith("RIGHT") && it.distanceMeters < 2f }
            val sideHint = when {
                !leftBusy -> "move slightly left"
                !rightBusy -> "move slightly right"
                else -> "slow down"
            }
            return "${centerObstacle.label} ahead at ${"%.1f".format(centerObstacle.distanceMeters)} meters, $sideHint."
        }

        val sideOnly = objects.minByOrNull { it.distanceMeters }
        return "Object on ${if (sideOnly?.gridZone?.endsWith("LEFT") == true) "left" else "right"}, proceed carefully at ${"%.1f".format(sideOnly?.distanceMeters ?: 0f)} meters."
    }
}
