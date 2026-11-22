package com.example.comicviewer.wm2

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max

/**
 * 확정: 같은 너비의 다른 장들에서도 비슷한 크기의 후보 상자가 반복될 때만 진짜 워터마크로 인정한다(한 장만 보면 말풍선·글자와 구분이 어렵다).
 * 이 클래스는 픽셀을 모른다. 이미지마다 후보 상자 목록만 받아 이번 페이지의 기록과 대조한다.
 */
class Wm2Confirmer {
    class Result(val confirmed: List<Wm2Detector.Box>, val others: Int, val firstTime: Boolean)
    private class Rec(val key: Long, val w: Int, val h: Int, val boxes: List<Wm2Detector.Box>)

    private val hist = ArrayList<Rec>()
    @Volatile var everConfirmed = false; private set
    val size: Int get() = synchronized(hist) { hist.size }

    fun reset() = synchronized(hist) { hist.clear(); everConfirmed = false }

    /** 이 이미지(key 로 구분, 같은 이미지를 다시 받으면 기록을 갈아 끼움)의 후보를 기록하고, 확정된 상자를 돌려준다 */
    fun confirm(key: Long, w: Int, h: Int, boxes: List<Wm2Detector.Box>): Result = synchronized(hist) {
        hist.removeAll { it.key == key }; hist.add(Rec(key, w, h, boxes)); if (hist.size > MAX_HIST) hist.removeAt(0)
        val others = hist.filter { it.key != key && it.w == w }
        if (others.size < MIN_OTHERS) return Result(emptyList(), others.size, false)
        val need = max(1, ceil(VOTE * others.size).toInt())
        val ok = boxes.filter { b -> others.count { o -> o.boxes.any { c -> similar(b, c) } } >= need }
        val first = ok.isNotEmpty() && !everConfirmed
        if (first) everConfirmed = true
        Result(ok, others.size, first)
    }

    companion object {
        const val MAX_HIST = 24
        const val MIN_OTHERS = 2      // 비교할 다른 장이 이만큼 있어야 판정(= 3장 이상)
        const val VOTE = 0.5          // 다른 장의 이 비율 이상에서 비슷한 상자가 나와야 확정
        const val TILE_N = 6          // 비슷한 작은 덩어리가 이만큼 많으면 전면 무늬 하나로 합침

        /** 크기가 비슷한가(위치는 따지지 않음: 흔들림·세로 제각각·완전 제각각도 인정) */
        fun similar(a: Wm2Detector.Box, b: Wm2Detector.Box, tol: Float = 0.35f): Boolean =
            a.tile == b.tile && abs(a.w - b.w) <= tol * max(a.w, b.w) + 6 && abs(a.h - b.h) <= tol * max(a.h, b.h) + 6

        fun tileMerge(bs: List<Wm2Detector.Box>, w: Int, h: Int): List<Wm2Detector.Box> {
            val small = bs.filter { !it.tile }
            if (small.size < TILE_N) return bs
            val areas = small.map { it.w.toLong() * it.h }.sorted(); val med = areas[areas.size / 2]
            if (areas.count { it >= 0.3 * med && it <= 3 * med } >= TILE_N) return listOf(Wm2Detector.Box(0, 0, w, h, true, 1f))
            return bs
        }
    }
}
