package com.example.comicviewer.wm2

import com.example.comicviewer.WmCodec
import com.example.comicviewer.WmFull
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * wm2 인식 엔진(인식 전용). 이미지를 가로채 후보 상자를 찾고(Wm2Detector), 같은 너비의 다른 장에서도 비슷한 크기의 상자가
 * 반복되는 것만 '확정'해 이미지에 빨간 상자로 그려 돌려준다. 지우기는 하지 않는다(기존 지우기 코드와 독립).
 * 확정 전(비교할 장이 모자람)에는 null(원본 그대로)을 돌려주고, 처음 확정되는 순간 onConfirmed 를 불러 앞서 지나간 이미지를 다시 받게 한다.
 */
class Wm2Engine(private val codec: WmCodec, private val det: Wm2Detector) {
    @Volatile var enabled = true
    @Volatile var logSink: ((String) -> Unit)? = null
    @Volatile var onConfirmed: (() -> Unit)? = null
    @Volatile var lastErr = ""
    @Volatile private var seen = 0
    @Volatile private var shown = 0
    @Volatile private var lastMs = 0L

    private class Rec(val key: Long, val w: Int, val h: Int, val boxes: List<Wm2Detector.Box>)
    private val lock = Any()
    private val hist = ArrayList<Rec>()      // 이 페이지에서 본 이미지의 후보 상자(최대 24장)
    private var confirmedOnce = false
    private val log = ArrayList<String>()

    private fun note(t: String) { synchronized(log) { log.add(t); if (log.size > 60) log.removeAt(0) }; try { logSink?.invoke(t) } catch (_: Throwable) { } }

    fun onPage() { synchronized(lock) { hist.clear(); confirmedOnce = false } }
    fun forget() = onPage()

    fun status(): String = synchronized(lock) {
        val ok = confirmedOnce
        if (!enabled) "wm2 꺼짐" else if (ok) "wm2 · 워터마크 확정 · 표시 ${shown}장 · 검사 ${seen}장 · ${lastMs}ms/장"
        else "wm2 · 찾는 중 · 검사 ${seen}장(같은 너비 ${hist.size}장, 3장 이상이면 판정) · ${lastMs}ms/장"
    }
    fun diag(): String = synchronized(log) { log.takeLast(12).joinToString("\n") }
    fun fullDiag(): String = "[wm2]\n" + status() + (if (lastErr.isNotEmpty()) "\n마지막 오류: $lastErr" else "") + "\n" + synchronized(log) { log.joinToString("\n") }

    /** 확정된 상자를 돌려준다(빈 목록 = 아직 확정 못 했거나 없음). 이미지 조건에 안 맞으면 null. 시험(JVM)에서도 이 함수를 직접 쓴다 */
    fun analyze(bytes: ByteArray): List<Wm2Detector.Box>? {
        val dim = codec.info(bytes) ?: run { lastErr = "이미지 해석 실패"; return null }
        val w = dim[0]; val h = dim[1]
        if (w < 350 || h < 100 || w > 4000 || h > 30000 || w.toLong() * h > 16_000_000L || w.toFloat() / h > 8f) return null
        val t0 = System.currentTimeMillis()
        val img = codec.small(bytes, 1) ?: run { lastErr = "디코드 실패"; return null }
        val boxes = tileMerge(det.detect(img), w, h)
        lastMs = System.currentTimeMillis() - t0; seen++
        val key = bytes.size * 31L + bytes.contentHashCode()
        var confirmed: List<Wm2Detector.Box> = emptyList(); var fire = false; var info: String
        synchronized(lock) {
            hist.removeAll { it.key == key }; hist.add(Rec(key, w, h, boxes)); if (hist.size > 24) hist.removeAt(0)
            val others = hist.filter { it.key != key && it.w == w }
            info = "#$seen ${w}x$h 후보 ${boxes.size}개 · 비교 ${others.size}장"
            if (others.size >= MIN_OTHERS) {
                val need = max(1, ceil(VOTE * others.size).toInt())
                confirmed = boxes.filter { b -> others.count { o -> o.boxes.any { c -> similar(b, c) } } >= need }
                info += " → 확정 ${confirmed.size}개"
                if (confirmed.isNotEmpty() && !confirmedOnce) { confirmedOnce = true; fire = true }
            } else info += " → 판정 대기"
        }
        note(info)
        if (fire) try { onConfirmed?.invoke() } catch (_: Throwable) { }
        return confirmed
    }

    /** 이미지 한 장 처리. 확정된 상자를 그려 돌려주고, 아직 확정 못 했거나 상자가 없으면 null */
    fun process(bytes: ByteArray, mime: String): ByteArray? {
        if (!enabled) return null
        try {
            val confirmed = analyze(bytes) ?: return null
            if (confirmed.isEmpty()) return null
            val full = codec.full(bytes) ?: return null
            try {
                for (b in confirmed) drawBox(full, b, full.w, full.h)
                shown++
                return full.encode(mime.contains("png"))
            } finally { full.close() }
        } catch (t: Throwable) { lastErr = (t.javaClass.simpleName + ":" + (t.message ?: "")).take(80); return null }
    }

    private fun drawBox(f: WmFull, b: Wm2Detector.Box, w: Int, h: Int) {
        val red = 0xFFFF0000.toInt(); val t = if (b.tile) 8 else 4
        val m = if (b.tile) 0 else 3
        val x0 = max(0, b.x - m); val y0 = max(0, b.y - m); val x1 = min(w, b.x + b.w + m); val y1 = min(h, b.y + b.h + m)
        fun fill(x: Int, y: Int, rw: Int, rh: Int) { if (rw <= 0 || rh <= 0) return; val a = IntArray(rw * rh) { red }; f.setRegion(x, y, rw, rh, a) }
        fill(x0, y0, x1 - x0, min(t, y1 - y0)); fill(x0, max(y0, y1 - t), x1 - x0, min(t, y1 - y0))
        fill(x0, y0, min(t, x1 - x0), y1 - y0); fill(max(x0, x1 - t), y0, min(t, x1 - x0), y1 - y0)
    }

    companion object {
        const val MIN_OTHERS = 2      // 비교할 다른 장이 이만큼 있어야 판정
        const val VOTE = 0.5          // 다른 장의 이 비율 이상에서 비슷한 상자가 나와야 확정
        const val TILE_N = 6          // 비슷한 작은 덩어리가 이만큼 많으면 전면 무늬

        /** 크기가 비슷한가(위치는 따지지 않음: 흔들림·세로 제각각·완전 제각각도 인정) */
        fun similar(a: Wm2Detector.Box, b: Wm2Detector.Box, tol: Float = 0.35f): Boolean =
            a.tile == b.tile && abs(a.w - b.w) <= tol * max(a.w, b.w) + 6 && abs(a.h - b.h) <= tol * max(a.h, b.h) + 6

        /** 비슷한 크기의 작은 덩어리가 6개 이상이면 전면 반복 무늬 하나로 합침 */
        fun tileMerge(bs: List<Wm2Detector.Box>, w: Int, h: Int): List<Wm2Detector.Box> {
            val small = bs.filter { !it.tile }
            if (small.size < TILE_N) return bs
            val areas = small.map { it.w.toLong() * it.h }.sorted(); val med = areas[areas.size / 2]
            if (areas.count { it >= 0.3 * med && it <= 3 * med } >= TILE_N) return listOf(Wm2Detector.Box(0, 0, w, h, true, 1f))
            return bs
        }
    }
}
