package com.example.comicviewer.wm2

import com.example.comicviewer.WmFull
import kotlin.math.max
import kotlin.math.min

/**
 * 확정된 워터마크 자리(모델의 확률 지도 = 마스크)를 주변 배경으로 메워 지운다.
 * 방식: 마스크(확률>ERASE_TH, 약간 팽창) 밖의 픽셀로 push-pull 다중 해상도 채우기 → 마스크 안쪽만 반복 평활.
 * 한 장씩 처리하며 다른 이미지나 이전 인식 결과에 의존하지 않는다(인식 → 제거).
 */
object Wm2Eraser {
    @JvmField var ERASE_TH = (System.getProperty("wm2.erth")?.toFloatOrNull() ?: 0.2f)   // 마스크로 볼 확률(상자 판정 0.5보다 낮춰 가장자리까지 덮음)
    @JvmField var GROW = (System.getProperty("wm2.grow")?.toIntOrNull() ?: 3)                // 마스크 팽창(원본 픽셀)
    const val PAD = 16           // 상자 바깥 여백(주변 배경 표본)

    fun erase(f: WmFull, boxes: List<Wm2Detector.Box>, p: FloatArray, w2: Int, h2: Int): Int {
        var n = 0
        for (b in boxes) {
            if (b.tile) continue   // 전면 반복 무늬는 지우지 않음(이미지 전체를 메우면 내용이 뭉개짐)
            val x0 = if (b.tile) 0 else max(0, b.x - PAD); val y0 = if (b.tile) 0 else max(0, b.y - PAD)
            val x1 = if (b.tile) f.w else min(f.w, b.x + b.w + PAD); val y1 = if (b.tile) f.h else min(f.h, b.y + b.h + PAD)
            val rw = x1 - x0; val rh = y1 - y0
            if (rw < 4 || rh < 4) continue
            val px = IntArray(rw * rh); f.getRegion(x0, y0, rw, rh, px)
            val mask = BooleanArray(rw * rh); var cnt = 0
            val bx0 = if (b.tile) 0 else b.x; val by0 = if (b.tile) 0 else b.y; val bx1 = if (b.tile) f.w else b.x + b.w; val by1 = if (b.tile) f.h else b.y + b.h
            for (y in 0 until rh) for (x in 0 until rw) {
                val gx = x0 + x; val gy = y0 + y
                if (gx < bx0 || gx >= bx1 || gy < by0 || gy >= by1) continue
                val sx = min(w2 - 1, gx / 2); val sy = min(h2 - 1, gy / 2)
                if (p[sy * w2 + sx] > ERASE_TH) { mask[y * rw + x] = true; cnt++ }
            }
            if (cnt == 0) continue
            val grown = grow(mask, rw, rh, GROW)
            fill(px, grown, rw, rh)
            f.setRegion(x0, y0, rw, rh, px); n++
        }
        return n
    }

    private fun grow(m: BooleanArray, w: Int, h: Int, r: Int): BooleanArray {
        var cur = m
        repeat(r) {
            val nx = cur.copyOf()
            for (y in 0 until h) for (x in 0 until w) if (cur[y * w + x]) {
                for (dy in -1..1) { val yy = y + dy; if (yy < 0 || yy >= h) continue; for (dx in -1..1) { val xx = x + dx; if (xx in 0 until w) nx[yy * w + xx] = true } }
            }
            cur = nx
        }
        return cur
    }

    /** 마스크 안 픽셀을 밖 픽셀로 채움(색 채널별) */
    private fun fill(px: IntArray, mask: BooleanArray, w: Int, h: Int) {
        val ch = Array(3) { c -> FloatArray(w * h) { i -> ((px[i] shr (16 - 8 * c)) and 255).toFloat() } }
        val wgt0 = FloatArray(w * h) { if (mask[it]) 0f else 1f }
        for (c in 0 until 3) pushPull(ch[c], wgt0, mask, w, h)
        for (i in px.indices) if (mask[i]) {
            val r = ch[0][i].toInt().coerceIn(0, 255); val g = ch[1][i].toInt().coerceIn(0, 255); val b = ch[2][i].toInt().coerceIn(0, 255)
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
    }

    private fun pushPull(v: FloatArray, wgt: FloatArray, mask: BooleanArray, w: Int, h: Int) {
        val vals = ArrayList<FloatArray>(); val wts = ArrayList<FloatArray>(); val ws = ArrayList<Int>(); val hs = ArrayList<Int>()
        vals.add(v.copyOf()); wts.add(wgt); ws.add(w); hs.add(h)
        while (ws.last() > 2 && hs.last() > 2 && vals.size < 9) {
            val pw = ws.last(); val ph = hs.last(); val cw = (pw + 1) / 2; val chh = (ph + 1) / 2
            val pv = vals.last(); val pwt = wts.last(); val cv = FloatArray(cw * chh); val cwt = FloatArray(cw * chh)
            for (y in 0 until chh) for (x in 0 until cw) {
                var sw = 0f; var sv = 0f
                for (dy in 0..1) { val yy = 2 * y + dy; if (yy >= ph) continue; for (dx in 0..1) { val xx = 2 * x + dx; if (xx >= pw) continue; val q = pwt[yy * pw + xx]; sw += q; sv += q * pv[yy * pw + xx] } }
                if (sw > 0f) { cv[y * cw + x] = sv / sw; cwt[y * cw + x] = min(1f, sw / 4f) }
            }
            vals.add(cv); wts.add(cwt); ws.add(cw); hs.add(chh)
        }
        // push: 가장 거친 곳부터, 비어 있는(가중치 부족) 칸을 거친 층에서 채움
        for (l in vals.size - 2 downTo 0) {
            val fw = ws[l]; val fh = hs[l]; val cw = ws[l + 1]; val cvv = vals[l + 1]; val fv = vals[l]; val fwt = wts[l]
            for (y in 0 until fh) for (x in 0 until fw) {
                val i = y * fw + x; val q = fwt[i]
                if (q >= 1f) continue
                val up = cvv[min(hs[l + 1] - 1, y / 2) * cw + min(cw - 1, x / 2)]
                fv[i] = q * fv[i] + (1f - q) * up
            }
        }
        val out = vals[0]
        // 마스크 안쪽만 평활(이웃 평균)해 층 경계의 각진 자국을 줄임
        repeat(40) {
            for (y in 0 until h) for (x in 0 until w) {
                val i = y * w + x; if (!mask[i]) continue
                var s = 0f; var n = 0
                if (x > 0) { s += out[i - 1]; n++ }; if (x < w - 1) { s += out[i + 1]; n++ }; if (y > 0) { s += out[i - w]; n++ }; if (y < h - 1) { s += out[i + w]; n++ }
                if (n > 0) out[i] = s / n
            }
        }
        System.arraycopy(out, 0, v, 0, v.size)
    }
}
