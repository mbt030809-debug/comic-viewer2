package com.example.comicviewer.wm2

import com.example.comicviewer.WmImg
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * 예시 마크 갤러리: 웹툰을 스크롤하다 나오는, 흰 또는 검은 배경 위에 원본 워터마크가 놓인 이미지.
 * 갤러리인지 알아보고(find), 그 안의 마크를 하나씩 잘라 모델 입력(makeRef)으로 만든다.
 * 배경은 행마다 가장 왼쪽 픽셀 색으로 본다(가로 띠마다 흰/검정이 달라도 됨). 마크 사이는 배경색으로 충분히(40픽셀 이상) 떨어져 있어야 한다.
 * 학습 쪽 infer_ref.py 의 split_gallery / make_ref 와 같은 계산이다.
 */
object Wm2Gallery {
    const val RH = 112; const val RW = 192          // 예시 입력 캔버스(0.5배 축소 좌표)
    private const val DIFF = 40                      // 배경과 이 이상 다르면 마크 픽셀
    private const val JOIN = 20                      // 이 거리(픽셀) 안이면 같은 마크로 묶음(정사각 팽창)
    private const val MAX_FRAC = 0.30f               // 마크 픽셀이 화면의 이 비율을 넘으면 갤러리가 아님(보통 만화 장은 행 배경 검사에서 이미 걸러짐)
    private const val MAX_MARKS = 8
    private const val MIN_PIX = 30

    class Mark(val x: Int, val y: Int, val w: Int, val h: Int, val bg: Float)

    /** 갤러리면 예시 마크 상자(위→아래), 아니면 null */
    fun find(img: WmImg): List<Mark>? {
        val w = img.w; val h = img.h
        if (h > 3000 || w < 100) return null
        val px = img.px
        // 행 배경은 흰색 또는 검정이어야 한다(웹툰 그림이 가장자리에서 시작하는 일반 장을 빨리 거름)
        val bg = IntArray(h * 3)
        for (y in 0 until h) {
            val p = px[y * w]; val r = p shr 16 and 255; val g = p shr 8 and 255; val b = p and 255
            val lo = max(r, max(g, b)); val hi = min(r, min(g, b))
            if (!(hi >= 235 || lo <= 20)) return null
            bg[y * 3] = r; bg[y * 3 + 1] = g; bg[y * 3 + 2] = b
        }
        val nb = BooleanArray(w * h); var cnt = 0
        for (y in 0 until h) {
            val br = bg[y * 3]; val bgc = bg[y * 3 + 1]; val bb = bg[y * 3 + 2]; val o = y * w
            for (x in 0 until w) {
                val p = px[o + x]
                if (max(abs((p shr 16 and 255) - br), max(abs((p shr 8 and 255) - bgc), abs((p and 255) - bb))) > DIFF) { nb[o + x] = true; cnt++ }
            }
            if (cnt > MAX_FRAC * w * h) return null
        }
        if (cnt < MIN_PIX) return null
        // 정사각 팽창(반지름 JOIN) = 적분 이미지로 "사각형 안에 마크 픽셀이 있나"
        val sw = w + 1; val ii = IntArray(sw * (h + 1))
        for (y in 0 until h) { var row = 0; for (x in 0 until w) { if (nb[y * w + x]) row++; ii[(y + 1) * sw + x + 1] = ii[y * sw + x + 1] + row } }
        val d = BooleanArray(w * h)
        for (y in 0 until h) { val y0 = max(0, y - JOIN); val y1 = min(h, y + JOIN + 1)
            for (x in 0 until w) { val x0 = max(0, x - JOIN); val x1 = min(w, x + JOIN + 1)
                if (ii[y1 * sw + x1] - ii[y0 * sw + x1] - ii[y1 * sw + x0] + ii[y0 * sw + x0] > 0) d[y * w + x] = true } }
        val lab = IntArray(w * h); val stack = IntArray(w * h); val out = ArrayList<Mark>(); var nl = 0
        for (i0 in 0 until w * h) {
            if (!d[i0] || lab[i0] != 0) continue
            nl++; var sp = 0; stack[sp++] = i0; lab[i0] = nl
            var x0 = Int.MAX_VALUE; var y0 = Int.MAX_VALUE; var x1 = -1; var y1 = -1; var c = 0
            while (sp > 0) {
                val i = stack[--sp]; val x = i % w; val y = i / w
                if (nb[i]) { c++; x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y) }
                if (x > 0 && d[i - 1] && lab[i - 1] == 0) { lab[i - 1] = nl; stack[sp++] = i - 1 }
                if (x < w - 1 && d[i + 1] && lab[i + 1] == 0) { lab[i + 1] = nl; stack[sp++] = i + 1 }
                if (y > 0 && d[i - w] && lab[i - w] == 0) { lab[i - w] = nl; stack[sp++] = i - w }
                if (y < h - 1 && d[i + w] && lab[i + w] == 0) { lab[i + w] = nl; stack[sp++] = i + w }
            }
            if (c < MIN_PIX || x1 < 0) continue
            out.add(Mark(x0, y0, x1 - x0 + 1, y1 - y0 + 1, (bg[y0 * 3] + bg[y0 * 3 + 1] + bg[y0 * 3 + 2]) / 3f))
            if (out.size > MAX_MARKS) return null
        }
        if (out.isEmpty()) return null
        out.sortBy { it.y }
        return out
    }

    /** 마크 하나를 모델 입력(CHW, 3*RH*RW)으로: 0.5배 면적평균 축소 → 112x192 캔버스(배경색) 가운데 */
    fun makeRef(img: WmImg, m: Mark): FloatArray {
        var nw = max(4, m.w / 2); var nh = max(4, m.h / 2)
        var a = areaResize(img, m, nw, nh)
        val f = min(1f, min((RW - 8f) / nw, (RH - 8f) / nh))
        if (f < 1f) { val w2 = max(4, (nw * f).toInt()); val h2 = max(4, (nh * f).toInt()); a = areaResize3(a, nw, nh, w2, h2); nw = w2; nh = h2 }
        val can = FloatArray(3 * RH * RW) { m.bg / 255f }
        val y0 = (RH - nh) / 2; val x0 = (RW - nw) / 2
        for (c in 0 until 3) for (y in 0 until nh) for (x in 0 until nw) can[c * RH * RW + (y0 + y) * RW + x0 + x] = a[c * nh * nw + y * nw + x]
        return can
    }

    /** 원본 일부(m)를 nw*nh 로 면적평균 축소(RGB 0..1, CHW) */
    private fun areaResize(img: WmImg, m: Mark, nw: Int, nh: Int): FloatArray {
        val src = FloatArray(3 * m.w * m.h)
        for (y in 0 until m.h) for (x in 0 until m.w) { val p = img.px[(m.y + y) * img.w + m.x + x]
            src[y * m.w + x] = (p shr 16 and 255) / 255f; src[m.w * m.h + y * m.w + x] = (p shr 8 and 255) / 255f; src[2 * m.w * m.h + y * m.w + x] = (p and 255) / 255f }
        return areaResize3(src, m.w, m.h, nw, nh)
    }

    private fun areaResize3(src: FloatArray, sw: Int, sh: Int, nw: Int, nh: Int): FloatArray {
        val out = FloatArray(3 * nw * nh)
        for (c in 0 until 3) for (y in 0 until nh) {
            val ya = y * sh.toFloat() / nh; val yb = (y + 1) * sh.toFloat() / nh
            for (x in 0 until nw) {
                val xa = x * sw.toFloat() / nw; val xb = (x + 1) * sw.toFloat() / nw
                var s = 0f; var ws = 0f
                var yy = ya.toInt()
                while (yy < yb && yy < sh) {
                    val wy = min(yy + 1f, yb) - max(yy.toFloat(), ya)
                    var xx = xa.toInt()
                    while (xx < xb && xx < sw) { val wx = min(xx + 1f, xb) - max(xx.toFloat(), xa); val wgt = wx * wy; s += src[c * sw * sh + yy * sw + xx] * wgt; ws += wgt; xx++ }
                    yy++
                }
                out[c * nw * nh + y * nw + x] = if (ws > 0) s / ws else 0f
            }
        }
        return out
    }
}
