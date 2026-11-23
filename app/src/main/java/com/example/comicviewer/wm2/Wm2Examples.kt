package com.example.comicviewer.wm2

import com.example.comicviewer.WmImg
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * 1단계(규칙): 페이지 속 단색(흰/검은) 여백 띠 안에서 "예시 워터마크" 후보 상자를 찾는다. 학습 없음.
 * 학습 쪽 s1_rule.py 규칙 v5 와 같은 계산이다(단색 띠 찾기 → 배경과 다른 픽셀을 14칸 팽창해 덩어리로 묶기 → 크기·채움·색 수 규칙으로 거르기).
 * 결과는 Wm2Stage1 신경망으로 한 번 더 거른다(hybrid).
 */
object Wm2Examples {
    private const val TOL = 14           // 흰/검정으로 보는 허용 오차
    private const val DIFF = 28          // 배경과 이만큼 다르면 마크 픽셀
    private const val JOIN = 14          // 팽창 반지름
    private const val BAND_MIN = 90      // 띠 최소 높이
    private const val ROW_FRAC = 0.45f   // 한 행에서 단색 픽셀이 이 비율을 넘어야 띠의 행

    class Band(val y0: Int, val y1: Int, val bg: Int)

    fun bands(img: WmImg): List<Band> {
        val w = img.w; val h = img.h; val px = img.px; val out = ArrayList<Band>()
        val isW = BooleanArray(h); val isB = BooleanArray(h)
        for (y in 0 until h) {
            var cw = 0; var cb = 0; val o = y * w
            for (x in 0 until w) { val p = px[o + x]; val g = ((p shr 16 and 255) + (p shr 8 and 255) + (p and 255)) / 3f; if (g >= 255 - TOL) cw++ else if (g <= TOL) cb++ }
            isW[y] = cw > ROW_FRAC * w; isB[y] = cb > ROW_FRAC * w
        }
        for ((bg, ok) in listOf(255 to isW, 0 to isB)) {
            var y = 0
            while (y < h) {
                if (ok[y]) { var y1 = y; while (y1 < h && ok[y1]) y1++; if (y1 - y >= BAND_MIN) out.add(Band(y, y1, bg)); y = y1 } else y++
            }
        }
        return out
    }

    /** 후보 상자(x,y,w,h 원본 좌표) 목록 */
    fun find(img: WmImg): List<IntArray> {
        val out = ArrayList<IntArray>()
        for (b in bands(img)) findInBand(img, b, out)
        return out
    }

    private fun findInBand(img: WmImg, band: Band, out: MutableList<IntArray>) {
        val W = img.w; val H = img.h; val bh = band.y1 - band.y0; val px = img.px
        val diff = BooleanArray(W * bh)
        for (y in 0 until bh) { val o = (band.y0 + y) * W
            for (x in 0 until W) { val p = px[o + x]
                if (max(abs((p shr 16 and 255) - band.bg), max(abs((p shr 8 and 255) - band.bg), abs((p and 255) - band.bg))) > DIFF) diff[y * W + x] = true } }
        // 정사각 팽창(반지름 JOIN)
        val sw = W + 1; val ii = IntArray(sw * (bh + 1))
        for (y in 0 until bh) { var row = 0; for (x in 0 until W) { if (diff[y * W + x]) row++; ii[(y + 1) * sw + x + 1] = ii[y * sw + x + 1] + row } }
        val d = BooleanArray(W * bh)
        for (y in 0 until bh) { val ya = max(0, y - JOIN); val yb = min(bh, y + JOIN + 1)
            for (x in 0 until W) { val xa = max(0, x - JOIN); val xb = min(W, x + JOIN + 1)
                if (ii[yb * sw + xb] - ii[ya * sw + xb] - ii[yb * sw + xa] + ii[ya * sw + xa] > 0) d[y * W + x] = true } }
        val lab = IntArray(W * bh); val stack = IntArray(W * bh); var nl = 0
        for (i0 in 0 until W * bh) {
            if (!d[i0] || lab[i0] != 0) continue
            nl++; var sp = 0; stack[sp++] = i0; lab[i0] = nl
            var x0 = Int.MAX_VALUE; var y0 = Int.MAX_VALUE; var x1 = -1; var y1 = -1; var c = 0
            val members = ArrayList<Int>()
            while (sp > 0) {
                val i = stack[--sp]; val x = i % W; val y = i / W
                if (diff[i]) { c++; members.add(i); x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y) }
                if (x > 0 && d[i - 1] && lab[i - 1] == 0) { lab[i - 1] = nl; stack[sp++] = i - 1 }
                if (x < W - 1 && d[i + 1] && lab[i + 1] == 0) { lab[i + 1] = nl; stack[sp++] = i + 1 }
                if (y > 0 && d[i - W] && lab[i - W] == 0) { lab[i - W] = nl; stack[sp++] = i - W }
                if (y < bh - 1 && d[i + W] && lab[i + W] == 0) { lab[i + W] = nl; stack[sp++] = i + W }
            }
            if (c < 120 || x1 < 0) continue
            val w = x1 + 1 - x0; val h = y1 + 1 - y0
            if (w > 0.5f * W || w < 0.06f * W || h < 18 || w.toLong() * h > 0.2f * W * H) continue
            if (x0 <= 2 || x1 + 1 >= W - 2 || y0 <= 1 || y1 + 1 >= bh - 1) continue         // 띠 가장자리에 닿음 = 잘린 것
            // 채움 비율·색 수(v4~v5)
            val m = BooleanArray(w * h); for (i in members) m[(i / W - y0) * w + (i % W - x0)] = true
            val fill = c.toFloat() / (w * h)
            val cnt = IntArray(512)
            for (i in members) { val p = px[(band.y0 + i / W) * W + i % W]; cnt[((p shr 16 and 255) / 48) * 64 + ((p shr 8 and 255) / 48) * 8 + (p and 255) / 48]++ }
            var nb = 0; for (v in cnt) if (v > 0.03f * c) nb++
            val closed = filledFraction(m, w, h) > 0.35f
            if (fill < 0.09f && !closed) continue                                             // 가는 선 몇 줄
            if (nb > 9) continue                                                              // 색이 연속적으로 번지는 것
            out.add(intArrayOf(x0, band.y0 + y0, w, h))
        }
    }

    /** 구멍을 메운 뒤 채워진 비율(scipy binary_fill_holes 와 같음: 가장자리에서 닿지 않는 배경 = 구멍) */
    private fun filledFraction(m: BooleanArray, w: Int, h: Int): Float {
        val reach = BooleanArray(w * h); val st = IntArray(w * h); var sp = 0
        fun push(i: Int) { if (!m[i] && !reach[i]) { reach[i] = true; st[sp++] = i } }
        for (x in 0 until w) { push(x); push((h - 1) * w + x) }
        for (y in 0 until h) { push(y * w); push(y * w + w - 1) }
        while (sp > 0) { val i = st[--sp]; val x = i % w; val y = i / w
            if (x > 0) push(i - 1); if (x < w - 1) push(i + 1); if (y > 0) push(i - w); if (y < h - 1) push(i + w) }
        var c = 0; for (i in 0 until w * h) if (m[i] || !reach[i]) c++
        return c.toFloat() / (w * h)
    }

    /** 예시 둘레(바깥 6칸) 픽셀 중앙값 = 이 예시의 배경색 */
    fun pageBg(img: WmImg, x: Int, y: Int, w: Int, h: Int): Float {
        val W = img.w; val H = img.h
        val ya = max(0, y - 6); val yb = min(H - 1, y + h + 6); val xa = max(0, x - 6); val xb = min(W - 1, x + w + 6)
        val v = ArrayList<Int>()
        fun add(px: Int) { v.add(px shr 16 and 255); v.add(px shr 8 and 255); v.add(px and 255) }
        for (xx in xa until xb) { add(img.px[ya * W + xx]); add(img.px[yb * W + xx]) }
        for (yy in ya until yb) { add(img.px[yy * W + xa]); add(img.px[yy * W + xb]) }
        v.sort()
        if (v.isEmpty()) return 255f
        return if (v.size % 2 == 1) v[v.size / 2].toFloat() else (v[v.size / 2 - 1] + v[v.size / 2]) / 2f
    }

    /** 후보 상자 → 모델 예시 입력(사방 4칸 넓혀 자름) */
    fun makeRef(img: WmImg, b: IntArray): FloatArray {
        val x0 = max(0, b[0] - 4); val y0 = max(0, b[1] - 4)
        val w = min(img.w - x0, b[2] + 8); val h = min(img.h - y0, b[3] + 8)
        return Wm2Gallery.makeRef(img, Wm2Gallery.Mark(x0, y0, w, h, pageBg(img, b[0], b[1], b[2], b[3])))
    }
}
