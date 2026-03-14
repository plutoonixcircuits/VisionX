package com.smartvisionassist.tracking

import com.smartvisionassist.ml.DetectedObject
import kotlin.math.hypot

class ObjectTracker {
    private data class Track(var id: Int, var x: Float, var y: Float, var age: Int, var label: String)

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    fun update(detections: List<DetectedObject>): List<DetectedObject> {
        val updated = mutableListOf<DetectedObject>()
        val usedTrackIds = mutableSetOf<Int>()

        for (det in detections) {
            val match = tracks
                .filter { it.label == det.label && it.id !in usedTrackIds }
                .minByOrNull { hypot((det.cx - it.x).toDouble(), (det.cy - it.y).toDouble()) }

            val assignedId = if (match != null && hypot((det.cx - match.x).toDouble(), (det.cy - match.y).toDouble()) < 80.0) {
                match.x = det.cx
                match.y = det.cy
                match.age = 0
                usedTrackIds += match.id
                match.id
            } else {
                val id = nextId++
                tracks += Track(id, det.cx, det.cy, 0, det.label)
                usedTrackIds += id
                id
            }

            updated += det.copy(id = assignedId)
        }

        tracks.forEach { it.age++ }
        tracks.removeAll { it.age > 8 }
        return updated
    }
}
