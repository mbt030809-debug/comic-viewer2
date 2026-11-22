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

    /** 이 이미지(key 로 구분, 같은 이미지를 다시 받으면 기록을 갈아 끼움)의 후보를 기록하고, 확정된 상자를 돌려준다.
     *  점수(예시와 얼마나 같은지, 0~1)가 HI 이상이면 이 장만 보고 확정. LO 이상이면 같은 너비의 다른 장들에서도 비슷한 크기의 후보가 반복될 때 확정. */
    fun confirm(key: Long, w: Int, h: Int, boxes: List<Wm2Detector.Box>): Result = synchronized(hist) {
        hist.removeAll { it.key == key }; hist.add(Rec(key, w, h, boxes)); if (hist.size > MAX_HIST) hist.removeAt(0)
        val others = hist.filter { it.key != key && it.w == w }
        val need = max(1, ceil(VOTE * others.size).toInt())
        val ok = boxes.filter { b ->
            b.score >= HI || (b.score >= LO && others.size >= MIN_OTHERS && others.count { o -> o.boxes.any { c -> c.score >= LO && similar(b, c) } } >= need)
        }
        val first = ok.isNotEmpty() && !everConfirmed
        if (first) everConfirmed = true
        Result(ok, others.size, first)
    }

    companion object {
        const val MAX_HIST = 24
        const val MIN_OTHERS = 2      // 비교할 다른 장이 이만큼 있어야 판정(= 3장 이상)
        const val VOTE = 0.34         // 다른 장의 이 비율 이상에서 비슷한 상자가 나와야 확정(점수가 LO~HI 일 때)
        const val HI = 0.90f          // 이 점수 이상이면 한 장만으로 확정
        const val LO = 0.80f          // 이 점수 미만은 버림

        /** 크기가 비슷한가(예시 번호·위치는 따지지 않음: 비슷한 예시끼리 번호가 뒤바뀔 수 있음: 흔들림·세로 제각각·완전 제각각도 인정) */
        fun similar(a: Wm2Detector.Box, b: Wm2Detector.Box, tol: Float = 0.6f): Boolean =
            abs(a.w - b.w) <= tol * max(a.w, b.w) + 6 && abs(a.h - b.h) <= tol * max(a.h, b.h) + 6
    }
}
