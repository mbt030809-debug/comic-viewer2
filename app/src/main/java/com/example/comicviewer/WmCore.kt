package com.example.comicviewer

import java.io.DataInputStream
import java.io.DataOutputStream
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/*
 * 워터마크 자동 감지·제거 핵심 알고리즘 (안드로이드 클래스에 의존하지 않는 순수 코드 → PC에서도 시험 가능)
 *
 * 원리: 반투명 워터마크는  I = J·(1-a) + a·L  (J=원본, a=투명도, L=로고 색).
 *  1) 감지: 같은 너비의 이미지 여러 장에서 '같은 자리에 같은 윤곽'이 반복되면 워터마크로 확정(그림은 장마다 다름)
 *  2) 학습: 워터마크 둘레가 한 색(흰/검정/단색)인 장은 원본 색 J를 알 수 있으므로,
 *           서로 다른 색 배경 위의 표본이 모이면 픽셀마다 I = J·(1-a) + v 를 회귀해 a, v(=a·L)를 구함
 *  3) 제거: 학습되면 J = (I - v)/(1 - a) 로 되돌림(그림 위에서도). 학습 전이라도 단색 배경 위는 배경색으로 채움
 *  4) 바뀜: 모델과 윤곽이 연속으로 안 맞으면 모델을 버리고 다시 감지
 */

/** 디코드된 이미지(ARGB) */
class WmImg(val w: Int, val h: Int, val px: IntArray)

/** 감지용 축소 윤곽(기울기) 지도. g=|dx|+|dy|, gx=|dx|, gy=|dy|. s=축소 배율, w/h=원본 크기 */
class WmThumb(val s: Int, val tw: Int, val th: Int, val g: FloatArray, val gx: FloatArray, val gy: FloatArray, val w: Int, val h: Int, val calm: Boolean = false, val info: String = "")

/** 감지 결과: 프레임(TL=위 기준, BL=아래 기준) 좌표의 상자(축소 칸 단위)와 윤곽 템플릿(|dx|, |dy| 각각) */
class WmDet(val frame: Int, val tx0: Int, val tyF: Int, val tbw: Int, val tbh: Int, val score: Float, val tgx: FloatArray, val tgy: FloatArray, val tref: Int, val locs: IntArray? = null)

/** 풀 이미지의 상자 주변 픽셀 조각 */
class WmReg(val px: IntArray, val x0: Int, val y0: Int, val imgH: Int)

object WmCore {
    const val TL = 0      // 위에서 같은 거리
    const val BL = 1      // 아래에서 같은 거리
    const val F25 = 2     // 높이 차이의 1/4 만큼 이동(워터마크 위치가 이미지 높이에 비례해 정해지는 경우)
    const val F50 = 3     // 가운데 기준
    const val F75 = 4
    const val FREE = 5    // 가로(x) 위치만 같고 세로(y) 위치는 이미지마다 다름: 장마다 템플릿으로 찾음
    val FRAMES = intArrayOf(TL, BL, F25, F50, F75)
    private fun fOf(frame: Int) = when (frame) { F25 -> 0.25f; F50 -> 0.5f; F75 -> 0.75f; else -> 0f }

    /** 프레임 좌표의 행(yf)이 높이 th 인 이미지의 몇 번째 행인지. tref = 프레임 기준 높이(감지 때 풀에서 가장 낮은 이미지의 칸 수) */
    fun rowOf(frame: Int, th: Int, yf: Int, tref: Int): Int = when (frame) {
        TL -> yf
        BL -> th - 1 - yf
        else -> yf + Math.round(fOf(frame) * (th - tref))
    }

    /** 모델 상자의 원본 해상도 위쪽 y (nyF: TL=위에서 거리, BL=아래에서 상자 아랫변까지 거리, 그 외=기준 높이 기준 위쪽 y) */
    fun nativeYOf(frame: Int, nyF: Int, nbh: Int, h: Int, tref: Int, s: Int): Int = when (frame) {
        TL -> nyF
        BL -> h - nyF - nbh
        else -> nyF + Math.round(fOf(frame) * (h - tref * s))
    }
    const val PRES_MIN = 0.10f     // 모델 윤곽(풋프린트 안, 방향별)과 이만큼 이상 닮으면 '워터마크가 있다'
    const val LEARN_D = 4000.0     // 배경 색 다양성(제곱합)이 이 이상이면 투명도 학습 가능
    const val REG_PAD = 24         // 감지 상자 바깥으로 읽는 여백(px): 둘레(고리) 검사용
    private const val VMIN = 4f

    fun scaleFor(w: Int): Int { var s = 1; while (s < 8 && w / (s * 2) >= 300) s *= 2; return s }

    fun gray(p: Int): Float = (((p shr 16) and 255) * 299 + ((p shr 8) and 255) * 587 + (p and 255) * 114) / 1000f

    private fun blur3(src: FloatArray, w: Int, h: Int): FloatArray {
        val t = FloatArray(w * h); val o = FloatArray(w * h)
        for (y in 0 until h) { val r = y * w
            for (x in 0 until w) t[r + x] = (src[r + max(x - 1, 0)] + src[r + x] + src[r + min(x + 1, w - 1)]) / 3f }
        for (y in 0 until h) { val r0 = max(y - 1, 0) * w; val r1 = y * w; val r2 = min(y + 1, h - 1) * w
            for (x in 0 until w) o[r1 + x] = (t[r0 + x] + t[r1 + x] + t[r2 + x]) / 3f }
        return o
    }

    /** 윤곽 강도: 3x3 평균 후 중앙 차분의 절댓값 합 */
    /** 색 윤곽: 채널(R,G,B)마다 윤곽을 구해 평균(세 채널의 윤곽 합)을 씀 — 붉은 워터마크처럼 밝기(회색)는 배경과 비슷해도 색이 다르면 잡힘 */
    fun colorGradMag(px: IntArray, w: Int, h: Int): FloatArray {
        val n = w * h; var out = FloatArray(n)
        for (c in 0..2) {
            val sh = 16 - 8 * c
            val g = gradMag(FloatArray(n) { ((px[it] shr sh) and 255).toFloat() }, w, h)
            if (c == 0) out = FloatArray(n) { g[it] / 3f } else for (i in 0 until n) out[i] += g[i] / 3f
        }
        return out
    }

    fun gradMag(gr: FloatArray, w: Int, h: Int): FloatArray {
        val bl = blur3(gr, w, h); val g = FloatArray(w * h)
        for (y in 1 until h - 1) { val r = y * w
            for (x in 1 until w - 1) g[r + x] = abs(bl[r + x + 1] - bl[r + x - 1]) + abs(bl[r + x + w] - bl[r + x - w]) }
        return g
    }

    fun makeThumb(small: WmImg, s: Int, nw: Int, nh: Int): WmThumb {
        val w = small.w; val h = small.h; val n = w * h
        val gx = FloatArray(n); val gy = FloatArray(n); val g = FloatArray(n)
        for (c in 0..2) {
            val sh = 16 - 8 * c
            val bl = blur3(FloatArray(n) { ((small.px[it] shr sh) and 255).toFloat() }, w, h)
            for (y in 1 until h - 1) { val r = y * w
                for (x in 1 until w - 1) { val a = abs(bl[r + x + 1] - bl[r + x - 1]); val b = abs(bl[r + x + w] - bl[r + x - w]); gx[r + x] += a / 3f; gy[r + x] += b / 3f } }
        }
        for (i in 0 until n) g[i] = gx[i] + gy[i]
        val sup = suppressFrames(g, gx, gy, w, h, s)
        var busy = 0; for (i in 0 until n) if (g[i] > CALM_G) busy++
        return WmThumb(s, w, h, g, gx, gy, nw, nh, busy <= CALM_FRAC * n, "s=$s 윤곽칸 ${busy * 100 / max(1, n)}% · 직선 ${sup[0]}개 제거(${sup[1]}%)")
    }

    private const val CALM_G = 12f       // '조용한 칸'으로 볼 윤곽 세기 상한
    private const val CALM_FRAC = 0.05f  // 윤곽 있는 칸이 이 비율 이하면 단색 위주 이미지

    /**
     * 컷 테두리·구분선처럼 길게 곧은 윤곽과 이미지 가장자리 윤곽을 지움: 이런 것은 이미지마다 같은 자리에 반복되지만 워터마크가 아님.
     * 워터마크 글자·로고의 획은 짧고 휘어 있어(곧은 선이 이미지 너비/높이의 1/4~1/3 이상 이어지지 않음) 남음.
     */
    private fun suppressFrames(g: FloatArray, gx: FloatArray, gy: FloatArray, w: Int, h: Int, s: Int): IntArray {
        val n = w * h
        val kill = BooleanArray(n)
        // 가장자리 여백
        val mg = max(2, 8 / s)
        for (y in 0 until h) for (x in 0 until w) if (x < mg || y < mg || x >= w - mg || y >= h - mg) kill[y * w + x] = true
        val lines = houghKill(g, w, h, kill)
        var killed = 0
        for (i in 0 until n) if (kill[i]) { g[i] = 0f; gx[i] = 0f; gy[i] = 0f; killed++ }
        return intArrayOf(lines, killed * 100 / max(1, n))
    }

    private const val HOUGH_G = 100f  // 직선 후보로 볼 강한 윤곽 세기: 검은 프레임·만화 선(대비 200 이상)만. 반투명 도장의 직선 띠는 이보다 약해 남김

    /**
     * 어떤 각도든 길게 이어진 곧은 선(컷 테두리·기울어진 구분선)을 찾아 지움(허프 변환).
     * 강한 윤곽 픽셀들이 같은 직선 위에서 끊김 없이(틈 2칸 이하) 이미지 너비의 15% 이상 이어질 때만 선으로 봄 → 우연히 일렬로 늘어선 그림 윤곽은 제외.
     */
    private fun houghKill(g: FloatArray, w: Int, h: Int, kill: BooleanArray): Int {
        var ne = 0; val ex = IntArray(w * h)
        for (i in 0 until w * h) if (g[i] > HOUGH_G) ex[ne++] = i
        if (ne < 100) return 0
        val nA = 180; val diag = kotlin.math.hypot(w.toDouble(), h.toDouble()).toInt() + 2; val nR = 2 * diag + 1
        val cosT = FloatArray(nA) { cos(Math.PI * it / nA).toFloat() }; val sinT = FloatArray(nA) { sin(Math.PI * it / nA).toFloat() }
        val acc = IntArray(nA * nR)
        for (k in 0 until ne) { val x = ex[k] % w; val y = ex[k] / w
            for (a in 0 until nA) { val r = Math.round(x * cosT[a] + y * sinT[a]) + diag; acc[a * nR + r]++ } }
        val minLen = max(30f, 0.15f * w)
        val minVotes = minLen.toInt()
        var tries = 0; var lines = 0
        val tv = FloatArray(ne); val idx = IntArray(ne)
        while (tries < 40 && lines < 16) {
            tries++
            var bi = 0; var bv = 0
            for (i in acc.indices) if (acc[i] > bv) { bv = acc[i]; bi = i }
            if (bv < minVotes) break
            val a0 = bi / nR; val r0 = bi % nR - diag
            // 이 직선 위(±1.5칸)의 윤곽 픽셀을 선을 따라 정렬해 끊김 없이 이어진 구간을 찾음
            var m = 0
            for (k in 0 until ne) { val x = ex[k] % w; val y = ex[k] / w
                if (abs(x * cosT[a0] + y * sinT[a0] - r0) <= 1.5f) { tv[m] = -x * sinT[a0] + y * cosT[a0]; idx[m] = ex[k]; m++ } }
            val ord = (0 until m).sortedBy { tv[it] }
            var st = 0; var found = false
            for (q in 1..ord.size) {
                if (q == ord.size || tv[ord[q]] - tv[ord[q - 1]] > 2.5f) {
                    if (ord.isNotEmpty() && tv[ord[q - 1]] - tv[ord[st]] >= minLen) {
                        found = true
                        for (u in st until q) { val pi = idx[ord[u]]; val px = pi % w; val py = pi / w
                            for (yy in max(0, py - 3)..min(h - 1, py + 3)) for (xx in max(0, px - 3)..min(w - 1, px + 3)) kill[yy * w + xx] = true }
                    }
                    st = q
                }
            }
            if (found) lines++
            for (da in -5..5) { val a = ((a0 + da) % nA + nA) % nA; for (dr in -5..5) { val r = bi % nR + dr; if (r in 0 until nR) acc[a * nR + r] = 0 } }
        }
        return lines
    }

    // ---------------------------------------------------------------- 감지
    private fun alignMap(t: WmThumb, a: FloatArray, frame: Int, hc: Int): FloatArray {
        val o = FloatArray(t.tw * hc)
        for (yf in 0 until hc) { val y = rowOf(frame, t.th, yf, hc); System.arraycopy(a, y * t.tw, o, yf * t.tw, t.tw) }
        return o
    }

    private fun boxSums(src: FloatArray, w: Int, h: Int, win: Int, ii: DoubleArray): FloatArray {
        val w1 = w + 1
        for (y in 0 until h) { var row = 0.0
            for (x in 0 until w) { row += src[y * w + x]; ii[(y + 1) * w1 + x + 1] = ii[y * w1 + x + 1] + row } }
        val half = win / 2; val out = FloatArray(w * h)
        for (y in half until h - half) for (x in half until w - half) {
            val x0 = x - half; val y0 = y - half; val x1 = x + half + 1; val y1 = y + half + 1
            out[y * w + x] = (ii[y1 * w1 + x1] - ii[y0 * w1 + x1] - ii[y1 * w1 + x0] + ii[y0 * w1 + x0]).toFloat()
        }
        return out
    }

    /** 이미지 쌍마다 창(win x win) 안의 윤곽 상관을 구해 평균: 같은 워터마크가 있는 자리만 높게 나옴 */
    private fun pairNcc(maps: Array<FloatArray>, w: Int, h: Int, win: Int): FloatArray {
        val k = maps.size; val n = (win * win).toFloat()
        val ii = DoubleArray((w + 1) * (h + 1))
        val s1 = Array(k) { boxSums(maps[it], w, h, win, ii) }
        val s2 = Array(k) { i -> val m = maps[i]; boxSums(FloatArray(w * h) { m[it] * m[it] }, w, h, win, ii) }
        val acc = FloatArray(w * h); val prod = FloatArray(w * h); var pairs = 0
        for (i in 0 until k) for (j in i + 1 until k) {
            val a = maps[i]; val b = maps[j]
            for (t in prod.indices) prod[t] = a[t] * b[t]
            val sab = boxSums(prod, w, h, win, ii)
            for (t in acc.indices) {
                val va = s2[i][t] - s1[i][t] * s1[i][t] / n; val vb = s2[j][t] - s1[j][t] * s1[j][t] / n
                if (va < VMIN * n || vb < VMIN * n) continue
                val c = (sab[t] - s1[i][t] * s1[j][t] / n) / sqrt(va * vb)
                if (c > 0f) acc[t] += c
            }
            pairs++
        }
        if (pairs > 0) for (t in acc.indices) acc[t] = acc[t] / pairs
        return acc
    }

    private fun dilate(b: BooleanArray, w: Int, h: Int, r: Int): BooleanArray {
        val t = BooleanArray(w * h); val o = BooleanArray(w * h)
        for (y in 0 until h) for (x in 0 until w) { var v = false; var xx = max(0, x - r); val xe = min(w - 1, x + r); while (xx <= xe) { if (b[y * w + xx]) { v = true; break }; xx++ }; t[y * w + x] = v }
        for (y in 0 until h) for (x in 0 until w) { var v = false; var yy = max(0, y - r); val ye = min(h - 1, y + r); while (yy <= ye) { if (t[yy * w + x]) { v = true; break }; yy++ }; o[y * w + x] = v }
        return o
    }
    private fun erode(b: BooleanArray, w: Int, h: Int, r: Int): BooleanArray {
        val inv = BooleanArray(w * h) { !b[it] }
        val d = dilate(inv, w, h, r)
        return BooleanArray(w * h) { !d[it] }
    }

    private fun blobIsMark(mapsX: Array<FloatArray>, mapsY: Array<FloatArray>, w: Int, x0: Int, y0: Int, x1: Int, y1: Int): Boolean {
        val bw = x1 - x0 + 1; val bh = y1 - y0 + 1; if (bw < 8 || bh < 8) return false
        val tgx = FloatArray(bw * bh); val tgy = FloatArray(bw * bh); val tmp = FloatArray(mapsX.size)
        for (j in 0 until bh) for (i in 0 until bw) { val at = (y0 + j) * w + x0 + i
            for (k in mapsX.indices) tmp[k] = mapsX[k][at]; tmp.sort(); tgx[j * bw + i] = tmp[tmp.size / 2]
            for (k in mapsY.indices) tmp[k] = mapsY[k][at]; tmp.sort(); tgy[j * bw + i] = tmp[tmp.size / 2] }
        return markShape(tgx, tgy, bw, bh)
    }

    /** 상관 지도에서 가장 그럴듯한 덩어리(워터마크)를 골라 상자와 템플릿을 만듦 */
    private fun pickBlob(m: FloatArray, mapsX: Array<FloatArray>, mapsY: Array<FloatArray>, w: Int, h: Int, frame: Int, f: Float = 1f): WmDet? { // h = 기준 높이(칸)
        val bin = BooleanArray(w * h) { m[it] > 0.30f * f }
        val cl = erode(dilate(bin, w, h, 2), w, h, 2)
        val lab = IntArray(w * h); val stack = IntArray(w * h)
        var best: WmDet? = null
        var id = 0
        for (st in 0 until w * h) {
            if (!cl[st] || lab[st] != 0) continue
            id++; var sp = 0; stack[sp++] = st; lab[st] = id
            var x0 = w; var y0 = h; var x1 = -1; var y1 = -1; var cnt = 0; var sum = 0.0
            while (sp > 0) {
                val p = stack[--sp]; val px = p % w; val py = p / w
                if (bin[p]) { cnt++; sum += m[p] }
                if (px < x0) x0 = px; if (px > x1) x1 = px; if (py < y0) y0 = py; if (py > y1) y1 = py
                if (px > 0 && cl[p - 1] && lab[p - 1] == 0) { lab[p - 1] = id; stack[sp++] = p - 1 }
                if (px < w - 1 && cl[p + 1] && lab[p + 1] == 0) { lab[p + 1] = id; stack[sp++] = p + 1 }
                if (py > 0 && cl[p - w] && lab[p - w] == 0) { lab[p - w] = id; stack[sp++] = p - w }
                if (py < h - 1 && cl[p + w] && lab[p + w] == 0) { lab[p + w] = id; stack[sp++] = p + w }
            }
            val bw = x1 - x0 + 1; val bh = y1 - y0 + 1
            blobStat[0]++
            if (cnt < 0.002f * w * h || bw > 0.7f * w || bh > 0.8f * h) { blobStat[1]++; continue }
            val asp = bw.toFloat() / bh; if (asp < 0.2f || asp > 4.5f) { blobStat[1]++; continue }
            val mean = (sum / max(cnt, 1)).toFloat(); if (mean < 0.33f * f) { blobStat[1]++; continue }
            val score = mean * sqrt(cnt.toFloat())
            if (best == null || score > best.score) {
                if (!blobIsMark(mapsX, mapsY, w, x0, y0, x1, y1)) { blobStat[2]++; shapeReasons.merge(shapeWhy.substringBefore('(').trim().ifEmpty { shapeWhy }, 1, Int::plus); continue }
                blobStat[3]++
                val pad = 3
                val bx0 = max(0, x0 - pad); val by0 = max(0, y0 - pad); val bx1 = min(w - 1, x1 + pad); val by1 = min(h - 1, y1 + pad)
                val tbw = bx1 - bx0 + 1; val tbh = by1 - by0 + 1
                val tgx = FloatArray(tbw * tbh); val tgy = FloatArray(tbw * tbh); val tmp = FloatArray(mapsX.size)
                for (j in 0 until tbh) for (i in 0 until tbw) {
                    val at = (by0 + j) * w + bx0 + i
                    for (k in mapsX.indices) tmp[k] = mapsX[k][at]
                    tmp.sort(); tgx[j * tbw + i] = tmp[tmp.size / 2]
                    for (k in mapsY.indices) tmp[k] = mapsY[k][at]
                    tmp.sort(); tgy[j * tbw + i] = tmp[tmp.size / 2]
                }
                best = WmDet(frame, bx0, by0, tbw, tbh, score, tgx, tgy, h)
            }
        }
        return best
    }

    /** (시험용) 프레임 좌표의 평균 윤곽 상관 지도 */
    fun nccMap(ts: List<WmThumb>, frame: Int): FloatArray {
        val tw = ts[0].tw; val hc = ts.minOf { it.th }
        return pairNcc(Array(ts.size) { alignMap(ts[it], ts[it].g, frame, hc) }, tw, hc, 11)
    }

    /**
     * 후보 윤곽 템플릿이 '마크(로고·글자·원 같은 모양)'인지 판단. 단순한 선·한두 방향뿐인 틀·가는 띠는 워터마크가 아님.
     *  1) 방향 다양성: 윤곽 방향(수직~수평, 6구간)이 3구간 이상에 각각 7% 이상 퍼져 있어야 하고, 수직/수평 두 구간이 90% 넘으면 안 됨(사각 틀·직선)
     *  2) 채움: 상자를 5x5 칸으로 나눠 윤곽이 있는 칸이 7칸 이상이어야 함(선 하나가 상자를 가로지르면 5칸뿐이라 제외)
     *  3) 한 방향으로 길게 늘어선 정도(주성분 비율)가 6 이하
     * tgx/tgy 는 각각 |가로 변화|, |세로 변화| 이므로 방향은 0°(수직 선)~90°(수평 선)로 접힌 값.
     */
    @Volatile var shapeWhy = ""   // 마지막 모양 판단에서 탈락한 이유(진단용)
    fun markShape(tgx: FloatArray, tgy: FloatArray, bw: Int, bh: Int): Boolean {
        val n = bw * bh; if (n < 100) { shapeWhy = "너무 작음"; return false }
        val mag = FloatArray(n) { tgx[it] + tgy[it] }
        val so = mag.copyOf(); so.sort()
        val thr = max(2f, 0.25f * so[(n * 0.95f).toInt().coerceIn(0, n - 1)])
        val hist = DoubleArray(6); var tot = 0.0
        val cell = DoubleArray(25); var sx = 0.0; var sy = 0.0; var sxx = 0.0; var syy = 0.0; var sxy = 0.0; var cnt = 0
        for (y in 0 until bh) for (x in 0 until bw) {
            val i = y * bw + x; if (mag[i] < thr) continue
            val ang = Math.toDegrees(Math.atan2(tgy[i].toDouble(), tgx[i].toDouble()))   // 0..90
            hist[(ang / 15.0).toInt().coerceIn(0, 5)] += mag[i].toDouble(); tot += mag[i]
            cell[(y * 5 / bh) * 5 + (x * 5 / bw)] += mag[i].toDouble()
            sx += x; sy += y; sxx += x.toDouble() * x; syy += y.toDouble() * y; sxy += x.toDouble() * y; cnt++
        }
        if (tot <= 0.0 || cnt < 30) { shapeWhy = "윤곽 거의 없음(${cnt}칸)"; return false }
        val spread = hist.count { it >= 0.07 * tot }
        val filled = cell.count { it >= 0.03 * tot }
        if (spread < 3) { shapeWhy = "방향 ${spread}종뿐(선 모양)"; return false }
        if ((hist[0] + hist[5]) >= 0.9 * tot) { shapeWhy = "수직·수평 ${(100 * (hist[0] + hist[5]) / tot).toInt()}%(사각 틀)"; return false }
        if (filled < 7) { shapeWhy = "상자 25칸 중 ${filled}칸만 채움(가는 띠)"; return false }
        val mx = sx / cnt; val my = sy / cnt
        val cxx = sxx / cnt - mx * mx; val cyy = syy / cnt - my * my; val cxy = sxy / cnt - mx * my
        val tr = cxx + cyy; val det = cxx * cyy - cxy * cxy
        val disc = Math.sqrt(max(0.0, tr * tr / 4 - det)); val l1 = tr / 2 + disc; val l2 = max(1e-6, tr / 2 - disc)
        if (Math.sqrt(l1 / l2) > 6.0) { shapeWhy = "길쭉함 %.1f".format(Math.sqrt(l1 / l2)); return false }
        shapeWhy = "통과(방향 ${spread}종·${filled}칸)"
        return true
    }

    @Volatile var dbgFree = ""
    @Volatile var lastPairMax = 0f; @Volatile var lastCandN = 0
    @Volatile var dbgDet = ""      // 마지막 고정 위치형 감지 결과 요약(진단용)
    private var blobStat = IntArray(4)   // [전체 덩어리, 크기·비율 탈락, 모양 탈락, 통과]
    private val shapeReasons = HashMap<String, Int>()
    @Volatile var lastMax = 0f   // 마지막 감지 시도에서 나온 가장 높은 반복 상관(진단용: 낮으면 같은 자리에 반복되는 것이 없다는 뜻)

    /** 같은 너비 이미지들의 축소 윤곽에서 반복되는 워터마크 찾기. 3장 이상 필요 */
    /** f<1 이면 약한 증거도 후보로 인정(호출한 쪽이 연속 시도에서 같은 후보가 또 나올 때만 확정) */
    fun detect(ts: List<WmThumb>, f: Float = 1f): WmDet? {
        val k = ts.size; if (k < 3) return null
        val tw = ts[0].tw; val hc = ts.minOf { it.th }
        if (tw < 40 || hc < 40) return null
        var best: WmDet? = null; var mxAll = 0f
        blobStat = IntArray(4); shapeReasons.clear()
        val calm = ts.indices.filter { ts[it].calm }
        for (frame in FRAMES) {
            val maps = Array(k) { alignMap(ts[it], ts[it].g, frame, hc) }
            val mx = Array(k) { alignMap(ts[it], ts[it].gx, frame, hc) }; val my = Array(k) { alignMap(ts[it], ts[it].gy, frame, hc) }
            val nm = pairNcc(maps, tw, hc, 11)
            for (v in nm) if (v > mxAll) mxAll = v
            var d = pickBlob(nm, mx, my, tw, hc, frame, f)
            // 단색 위주(조용한) 이미지가 2장 이상이면 그 이미지들끼리의 반복을 먼저 봄: 그 자리의 윤곽은 워터마크뿐이라 그림·프레임에 안 흔들림
            if (calm.size >= 2) {
                val cm = pairNcc(Array(calm.size) { maps[calm[it]] }, tw, hc, 11)
                val dc = pickBlob(cm, Array(calm.size) { mx[calm[it]] }, Array(calm.size) { my[calm[it]] }, tw, hc, frame, f)
                if (dc != null) d = dc
            }
            if (d != null && (best == null || d.score > best.score)) best = d
        }
        lastMax = mxAll
        dbgDet = "고정위치형(f=%.1f): 반복 최고 %.2f · 덩어리 %d(크기·비율 탈락 %d · 모양 탈락 %d%s · 통과 %d) · 단색이미지 %d장 → %s".format(f, mxAll, blobStat[0], blobStat[1], blobStat[2], if (shapeReasons.isEmpty()) "" else ": " + shapeReasons.entries.joinToString(", ") { it.key + " " + it.value }, blobStat[3], calm.size, if (best != null) "후보 있음" else "후보 없음")
        return best
    }

    // ---------------------------------------------------------------- 확인
    private fun ncc(m: WmModel, t: WmThumb, dx: Int, dy: Int): Float {
        var n = 0; var sa = 0.0; var sb = 0.0; var saa = 0.0; var sbb = 0.0; var sab = 0.0
        for (j in 0 until m.tbh) {
            val yf = m.tyF + j + dy; val y = rowOf(m.frame, t.th, yf, m.tref)
            if (y < 0 || y >= t.th) return -1f
            for (i in 0 until m.tbw) {
                if (!m.tsup[j * m.tbw + i]) continue
                val x = m.tx0 + i + dx; if (x < 0 || x >= t.tw) return -1f
                val k = y * t.tw + x; val q = j * m.tbw + i
                val a1 = t.gx[k].toDouble(); val b1 = m.tgx[q].toDouble(); val a2 = t.gy[k].toDouble(); val b2 = m.tgy[q].toDouble()
                sa += a1 + a2; sb += b1 + b2; saa += a1 * a1 + a2 * a2; sbb += b1 * b1 + b2 * b2; sab += a1 * b1 + a2 * b2; n += 2
            }
        }
        if (n < 40) return 0f
        val va = saa - sa * sa / n; val vb = sbb - sb * sb / n
        if (va < VMIN * n || vb < VMIN * n) return 0f
        return ((sab - sa * sb / n) / sqrt(va * vb)).toFloat()
    }

    /** 반환 [일치도] (±2칸 어긋남 허용) */
    fun presence(m: WmModel, t: WmThumb): Float = locate(m, t)[0]

    /** [일치도, dx(칸), 위치(FREE 는 상자 위쪽 행, 그 외는 dy)] */
    fun locate(m: WmModel, t: WmThumb): FloatArray {
        if (t.tw != m.tw) return floatArrayOf(-1f, 0f, 0f)
        if (m.frame == FREE) { val r = findFree(m.tgx, m.tgy, m.tsup, m.tbw, m.tbh, m.tx0, t); return floatArrayOf(r[0], r[1], r[2]) }
        var best = -1f; var bx = 0; var by = 0
        for (dy in -2..2) for (dx in -2..2) { val c = ncc(m, t, dx, dy); if (c > best) { best = c; bx = dx; by = dy } }
        return floatArrayOf(best, bx.toFloat(), by.toFloat())
    }

    private fun nccFree(tgx: FloatArray, tgy: FloatArray, tsup: BooleanArray, tbw: Int, tbh: Int, t: WmThumb, tx: Int, ty: Int): Float {
        if (tx < 0 || ty < 0 || tx + tbw > t.tw || ty + tbh > t.th) return -1f
        var n = 0; var sa = 0.0; var sb = 0.0; var saa = 0.0; var sbb = 0.0; var sab = 0.0
        for (j in 0 until tbh) { val r = (ty + j) * t.tw + tx; val q0 = j * tbw
            for (i in 0 until tbw) {
                if (!tsup[q0 + i]) continue
                val a1 = t.gx[r + i].toDouble(); val b1 = tgx[q0 + i].toDouble(); val a2 = t.gy[r + i].toDouble(); val b2 = tgy[q0 + i].toDouble()
                sa += a1 + a2; sb += b1 + b2; saa += a1 * a1 + a2 * a2; sbb += b1 * b1 + b2 * b2; sab += a1 * b1 + a2 * b2; n += 2
            } }
        if (n < 40) return 0f
        val va = saa - sa * sa / n; val vb = sbb - sb * sb / n
        if (va < VMIN * n || vb < VMIN * n) return 0f
        return ((sab - sa * sb / n) / sqrt(va * vb)).toFloat()
    }

    /** 세로 전체를 훑어(성기게 → 촘촘히) 템플릿이 가장 잘 맞는 자리를 찾음. 반환 [일치도, dx, 위쪽 행] */
    fun findFree(tgx: FloatArray, tgy: FloatArray, tsup: BooleanArray, tbw: Int, tbh: Int, tx0: Int, t: WmThumb): FloatArray {
        val maxY = t.th - tbh; if (maxY < 0) return floatArrayOf(-1f, 0f, 0f)
        val cs = ArrayList<FloatArray>()
        var y = 0
        while (y <= maxY) { cs.add(floatArrayOf(nccFree(tgx, tgy, tsup, tbw, tbh, t, tx0, y), y.toFloat())); y += 3 }
        cs.sortByDescending { it[0] }
        var best = -1f; var bx = 0; var by = 0
        for (k in 0 until min(3, cs.size)) {
            val cy = cs[k][1].toInt()
            for (yy in max(0, cy - 3)..min(maxY, cy + 3)) for (dx in -2..2) { val c = nccFree(tgx, tgy, tsup, tbw, tbh, t, tx0 + dx, yy); if (c > best) { best = c; bx = dx; by = yy } }
        }
        return floatArrayOf(best, bx.toFloat(), by.toFloat())
    }

    // ---------------------------------------------------------------- 세로 위치가 제각각인 워터마크 감지
    private class Coarse(val w: Int, val h: Int, val g: FloatArray, val s1: FloatArray, val s2: FloatArray)
    private class PR(val score: Float, val x: Int, val y: Int, val dy: Int)

    private fun coarseOf(t: WmThumb, c: Int, win: Int): Coarse {
        val w = t.tw / c; val h = t.th / c; val g = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) { var sum = 0f; for (yy in 0 until c) { val r = (y * c + yy) * t.tw + x * c; for (xx in 0 until c) sum += t.g[r + xx] }; g[y * w + x] = sum / (c * c) }
        val ii = DoubleArray((w + 1) * (h + 1))
        val s1 = boxSums(g, w, h, win, ii); val s2 = boxSums(FloatArray(w * h) { g[it] * g[it] }, w, h, win, ii)
        return Coarse(w, h, g, s1, s2)
    }

    /** B 의 행 = A 의 행 + dy 로 겹쳐 보며(가로 위치는 같다고 보고) 창 상관이 가장 높은 곳을 찾음. map!=null 이면 dy==mapDy 일 때 상관 지도를 채움 */
    private fun searchPair(a: Coarse, b: Coarse, win: Int, mapDy: Int = Int.MIN_VALUE, map: FloatArray? = null): PR {
        val half = win / 2; val w = a.w; val n = (win * win).toFloat()
        var best = PR(-1f, 0, 0, 0)
        val minOv = win + 12
        val prod = FloatArray(w * max(a.h, b.h)); val w1 = w + 1
        val ii = DoubleArray(w1 * (max(a.h, b.h) + 1))
        for (dy in (-(a.h - minOv))..(b.h - minOv)) {
            if (map != null && dy != mapDy) continue
            val ya = max(half, half - dy); val yb = min(a.h - half, b.h - dy - half)
            if (yb - ya < 8) continue
            val r0 = ya - half; val r1 = yb + half; val rows = r1 - r0
            for (y in 0 until rows) { var row = 0.0
                for (x in 0 until w) { row += a.g[(r0 + y) * w + x] * b.g[(r0 + y + dy) * w + x]; ii[(y + 1) * w1 + x + 1] = ii[y * w1 + x + 1] + row } }
            for (y in ya until yb) for (x in half until w - half) {
                val ia = y * w + x; val ib = (y + dy) * w + x
                val va = a.s2[ia] - a.s1[ia] * a.s1[ia] / n; val vb = b.s2[ib] - b.s1[ib] * b.s1[ib] / n
                if (va < VMIN * n || vb < VMIN * n) continue
                val yy0 = y - half - r0; val yy1 = y + half + 1 - r0
                val sab = (ii[yy1 * w1 + x + half + 1] - ii[yy0 * w1 + x + half + 1] - ii[yy1 * w1 + x - half] + ii[yy0 * w1 + x - half]).toFloat()
                val c = (sab - a.s1[ia] * b.s1[ib] / n) / sqrt(va * vb)
                if (map != null) map[ia] = c
                if (c > best.score) best = PR(c, x, y, dy)
            }
        }
        return best
    }

    /** 가로 위치(x)는 같고 세로 위치는 제각각인 반복 워터마크 감지(같은 그림이 2장 이상에서 같은 x 에 나타남). 풀 3장 이상 */
    fun detectFree(ts: List<WmThumb>, f: Float = 1f): WmDet? {
        if (ts.size < 3) { dbgFree = "이미지 3장 미만"; return null }
        val c = 3; val win = 9
        val calmIdx = ts.indices.filter { ts[it].calm }
        val spread = if (ts.size <= 4) ts.indices.toList() else listOf(0, ts.size / 3, 2 * ts.size / 3, ts.size - 1)
        val sel = if (ts.size > 4 && calmIdx.size >= 2) (calmIdx.take(3) + spread).distinct().take(4).sorted() else spread
        if (sel.any { ts[it].tw / c < 40 || ts[it].th / c < 40 }) { dbgFree = "이미지가 너무 작음"; return null }
        val co = HashMap<Int, Coarse>(); for (i in sel) co[i] = coarseOf(ts[i], c, win)
        val prs = HashMap<Long, PR>()
        for (a in sel.indices) for (b in a + 1 until sel.size) prs[a * 10L + b] = searchPair(co[sel[a]]!!, co[sel[b]]!!, win)
        lastPairMax = prs.values.maxOf { it.score }; lastMax = max(lastMax, lastPairMax)
        // 후보 쌍: 세 장이 서로 일관(dy 가 더해지고 x 가 같음)하면 우연이 아님, 아니면 한 쌍이 아주 강하게 닮은 경우
        val cands = ArrayList<Triple<Float, Int, Int>>()
        for (a in sel.indices) for (b in a + 1 until sel.size) for (k in b + 1 until sel.size) {
            val ab = prs[a * 10L + b]!!; val bk = prs[b * 10L + k]!!; val ak = prs[a * 10L + k]!!
            val mn = minOf(ab.score, bk.score, ak.score)
            if (mn >= 0.3f * f && abs(ab.dy + bk.dy - ak.dy) <= 1 && abs(ab.x - bk.x) <= 2 && abs(ab.x - ak.x) <= 2) cands.add(Triple(mn + 1f, a, b))
        }
        for (a in sel.indices) for (b in a + 1 until sel.size) { val p = prs[a * 10L + b]!!; val both = ts[sel[a]].calm && ts[sel[b]].calm
            if (p.score >= 0.75f || (both && p.score >= 0.5f * f)) cands.add(Triple(p.score + (if (both) 1f else 0f), a, b)) }
        lastCandN = cands.size
        cands.sortByDescending { it.first }
        // 이미지 전체가 거의 같은 쌍(중복 이미지)은 상자가 너무 커서 실패하므로 다음 후보로
        for (cd in cands.take(8)) { val r = tryPair(ts, sel, co, prs, cd.second, cd.third, c, win); if (r != null) return r }
        dbgFree = "반복 후보 없음(쌍 최고 상관 %.2f, 일관된 삼중 후보 %d개)".format(lastPairMax, lastCandN); return null
    }

    private fun tryPair(ts: List<WmThumb>, sel: List<Int>, co: HashMap<Int, Coarse>, prs: HashMap<Long, PR>, bi: Int, bj: Int, c: Int, win: Int): WmDet? {
        val A = ts[sel[bi]]; val B = ts[sel[bj]]; val pr = prs[bi * 10L + bj]!!
        // 상관 지도로 상자 범위 찾기(최고점에서 이어지는 영역)
        val ca = co[sel[bi]]!!; val map = FloatArray(ca.w * ca.h)
        searchPair(ca, co[sel[bj]]!!, win, pr.dy, map)
        val bin = BooleanArray(map.size) { map[it] > 0.3f }
        val seen = BooleanArray(map.size); val st = IntArray(map.size); var sp = 0
        var x0 = pr.x; var x1 = pr.x; var y0 = pr.y; var y1 = pr.y
        val start = pr.y * ca.w + pr.x
        if (!bin[start]) bin[start] = true
        st[sp++] = start; seen[start] = true
        while (sp > 0) { val p = st[--sp]; val px = p % ca.w; val py = p / ca.w
            x0 = min(x0, px); x1 = max(x1, px); y0 = min(y0, py); y1 = max(y1, py)
            for ((dx, dy) in arrayOf(-1 to 0, 1 to 0, 0 to -1, 0 to 1)) { val nx = px + dx; val ny = py + dy
                if (nx in 0 until ca.w && ny in 0 until ca.h && bin[ny * ca.w + nx] && !seen[ny * ca.w + nx]) { seen[ny * ca.w + nx] = true; st[sp++] = ny * ca.w + nx } } }
        val pad = 3
        val tx0 = max(0, (x0 - win / 2) * c - pad); val txe = min(A.tw, (x1 + win / 2 + 1) * c + pad)
        val tyA = max(0, (y0 - win / 2) * c - pad); val tye = min(A.th, (y1 + win / 2 + 1) * c + pad)
        val tbw = txe - tx0; val tbh = tye - tyA
        run { val asp = tbw.toFloat() / max(1, tbh); if (asp < 0.2f || asp > 4.5f) { dbgFree = "후보 상자 비율 이상(가로세로 비 %.1f)".format(asp); return null } }
        if (tbw < 12 || tbh < 12 || tbw > 0.8f * A.tw || tbh > 0.8f * A.th) { dbgFree = "후보 상자 크기 이상(${tbw * A.s}x${tbh * A.s}px, 이미지 ${A.w}x${A.h})"; return null }
        val tyB = tyA + pr.dy * c
        if (tyB < 0 || tyB + tbh > B.th) { dbgFree = "두 번째 이미지에서 범위 밖"; return null }
        // 초기 템플릿: 두 장의 공통(작은 쪽) 윤곽
        val tgx = FloatArray(tbw * tbh); val tgy = FloatArray(tbw * tbh)
        for (j in 0 until tbh) for (i in 0 until tbw) { val ka = (tyA + j) * A.tw + tx0 + i; val kb = (tyB + j) * B.tw + tx0 + i
            tgx[j * tbw + i] = min(A.gx[ka], B.gx[kb]); tgy[j * tbw + i] = min(A.gy[ka], B.gy[kb]) }
        fun supOf(gx: FloatArray, gy: FloatArray): BooleanArray {
            val tl = FloatArray(gx.size) { gx[it] + gy[it] }; val so = tl.copyOf(); so.sort()
            val thr = max(6f, 0.2f * so[(so.size * 0.95f).toInt().coerceIn(0, so.size - 1)])
            return dilate(BooleanArray(tl.size) { tl[it] > thr }, tbw, tbh, 2)
        }
        var tsup = supOf(tgx, tgy)
        // 모든 풀 이미지에서 위치 찾기 → 멤버
        val locs = IntArray(ts.size) { -1 }; val nccs = FloatArray(ts.size)
        for (i in ts.indices) { if (ts[i].tw != A.tw) continue; val r = findFree(tgx, tgy, tsup, tbw, tbh, tx0, ts[i]); if (r[0] >= 0.25f && r[1] == 0f) { locs[i] = r[2].toInt(); nccs[i] = r[0] } else if (r[0] >= 0.25f) { locs[i] = r[2].toInt(); nccs[i] = r[0] } }
        val members = ts.indices.filter { locs[it] >= 0 }
        if (members.size < 2) { dbgFree = "같은 자리에서 확인된 이미지 2장 미만"; return null }
        // 상관 영역은 워터마크보다 작게 잡히기 쉬우므로, 멤버들의 윤곽 중앙값으로 실제 범위(풋프린트)를 다시 찾아 상자를 넓힘
        var bx = tx0; var by = tyA; var bw2 = tbw; var bh2 = tbh
        run {
            val ex = max(tbw, tbh) / 2
            val exY = minOf(ex, members.minOf { locs[it] }, members.minOf { ts[it].th - locs[it] - tbh })
            val exX = minOf(ex, tx0, A.tw - tx0 - tbw)
            if (exY < 2 || exX < 2) return@run
            val ww = tbw + 2 * exX; val hh = tbh + 2 * exY
            val med = FloatArray(ww * hh); val tmp = FloatArray(members.size)
            for (j in 0 until hh) for (i in 0 until ww) {
                for ((k, mi) in members.withIndex()) { val q = (locs[mi] - exY + j) * ts[mi].tw + tx0 - exX + i; tmp[k] = ts[mi].g[q] }
                tmp.sort(); med[j * ww + i] = if (members.size >= 3) tmp[tmp.size / 2] else tmp[0]
            }
            val so = med.copyOf(); so.sort()
            val thr = max(6f, 0.2f * so[(so.size * 0.95f).toInt().coerceIn(0, so.size - 1)])
            var mask = erode(dilate(BooleanArray(med.size) { med[it] > thr }, ww, hh, 2), ww, hh, 1)
            // 원래 상자와 겹치는 가장 큰 덩어리의 범위
            val lab = IntArray(med.size); val st = IntArray(med.size)
            var bestCnt = 0; var n0 = ww; var n1 = -1; var m0 = hh; var m1 = -1
            for (p0 in mask.indices) {
                if (!mask[p0] || lab[p0] != 0) continue
                var sp = 0; st[sp++] = p0; lab[p0] = 1
                var cx0 = ww; var cx1 = -1; var cy0 = hh; var cy1 = -1; var cnt = 0
                while (sp > 0) { val p = st[--sp]; val px = p % ww; val py = p / ww; cnt++
                    cx0 = min(cx0, px); cx1 = max(cx1, px); cy0 = min(cy0, py); cy1 = max(cy1, py)
                    if (px > 0 && mask[p - 1] && lab[p - 1] == 0) { lab[p - 1] = 1; st[sp++] = p - 1 }
                    if (px < ww - 1 && mask[p + 1] && lab[p + 1] == 0) { lab[p + 1] = 1; st[sp++] = p + 1 }
                    if (py > 0 && mask[p - ww] && lab[p - ww] == 0) { lab[p - ww] = 1; st[sp++] = p - ww }
                    if (py < hh - 1 && mask[p + ww] && lab[p + ww] == 0) { lab[p + ww] = 1; st[sp++] = p + ww } }
                if (cnt > bestCnt) { bestCnt = cnt; n0 = cx0; n1 = cx1; m0 = cy0; m1 = cy1 }
            }
            if (bestCnt < 30) return@run
            val padn = 3
            val nx0 = max(0, n0 - padn); val nx1 = min(ww - 1, n1 + padn); val ny0 = max(0, m0 - padn); val ny1 = min(hh - 1, m1 + padn)
            val nw = nx1 - nx0 + 1; val nh = ny1 - ny0 + 1
            if (nw < 12 || nh < 12 || nw > 0.9f * A.tw || nh > 0.9f * A.th) return@run
            val dyShift = ny0 - exY
            for (mi in members) locs[mi] += dyShift
            bx = tx0 - exX + nx0; by = tyA + dyShift; bw2 = nw; bh2 = nh
        }
        val tgx2 = FloatArray(bw2 * bh2); val tgy2 = FloatArray(bw2 * bh2)
        run {
            val tmp = FloatArray(members.size)
            for (j in 0 until bh2) for (i in 0 until bw2) {
                for ((k, mi) in members.withIndex()) tmp[k] = ts[mi].gx[(locs[mi] + j) * ts[mi].tw + bx + i]; tmp.sort(); tgx2[j * bw2 + i] = if (members.size >= 3) tmp[tmp.size / 2] else tmp[0]
                for ((k, mi) in members.withIndex()) tmp[k] = ts[mi].gy[(locs[mi] + j) * ts[mi].tw + bx + i]; tmp.sort(); tgy2[j * bw2 + i] = if (members.size >= 3) tmp[tmp.size / 2] else tmp[0]
            }
        }
        // supOf 는 원래 크기 기준이므로 새 크기로 다시 계산(아래 WmDet 로 전달, 모델 생성 때 tsup 을 템플릿에서 다시 구함)
        if (!markShape(tgx2, tgy2, bw2, bh2)) { dbgFree = "모양 탈락: " + shapeWhy + " (상자 ${bw2 * A.s}x${bh2 * A.s}px)"; return null }
        val mean = members.map { nccs[it] }.average().toFloat()
        return WmDet(FREE, bx, by, bw2, bh2, mean * sqrt(members.size.toFloat()), tgx2, tgy2, 0, locs)
    }


    /**
     * 상자 조각(원본 해상도)을 축소 칸으로 평균내 윤곽을 만들고 모델 템플릿과의 일치도를 구함(어긋남 탐색 없음).
     * 지우기 전/후를 같은 방식으로 비교해, 지운 뒤에도 윤곽이 남아 있거나 오히려 늘면(=잘못 맞춘 것) 지우기를 취소하는 데 씀
     */
    fun regionPresence(m: WmModel, region: IntArray): Float {
        val s = m.s; val cw = min(m.nbw / s, m.tbw); val ch = min(m.nbh / s, m.tbh)
        if (cw < 8 || ch < 8) return 0f
        val bls = Array(3) { c ->
            val sh = 16 - 8 * c; val gr = FloatArray(cw * ch)
            for (cy in 0 until ch) for (cx in 0 until cw) {
                var sum = 0f
                for (yy in 0 until s) { val r = (cy * s + yy) * m.nbw + cx * s; for (xx in 0 until s) sum += ((region[r + xx] shr sh) and 255) }
                gr[cy * cw + cx] = sum / (s * s)
            }
            blur3(gr, cw, ch)
        }
        var n = 0; var sa = 0.0; var sb = 0.0; var saa = 0.0; var sbb = 0.0; var sab = 0.0
        for (y in 1 until ch - 1) for (x in 1 until cw - 1) {
            val q = y * m.tbw + x; if (!m.tsup[q]) continue
            val k = y * cw + x
            var a1 = 0.0; var a2 = 0.0
            for (c in 0..2) { val bl = bls[c]; val v1 = abs(bl[k + 1] - bl[k - 1]).toDouble(); val v2 = abs(bl[k + cw] - bl[k - cw]).toDouble(); a1 += v1 / 3.0; a2 += v2 / 3.0 }
            val b1 = m.tgx[q].toDouble(); val b2 = m.tgy[q].toDouble()
            sa += a1 + a2; sb += b1 + b2; saa += a1 * a1 + a2 * a2; sbb += b1 * b1 + b2 * b2; sab += a1 * b1 + a2 * b2; n += 2
        }
        if (n < 40) return 0f
        val va = saa - sa * sa / n; val vb = sbb - sb * sb / n
        if (va < VMIN * n || vb < VMIN * n) return 0f
        return ((sab - sa * sb / n) / sqrt(va * vb)).toFloat()
    }

    // ---------------------------------------------------------------- 모델 만들기
    /** 둘레(상자 바깥 4~12px)가 한 색이면 그 색(RGB) 반환, 아니면 null. px=상자+여백 조각(pw x ph), 상자는 (bx,by,bw,bh) */
    fun ringFlat(px: IntArray, pw: Int, ph: Int, bx: Int, by: Int, bw: Int, bh: Int): IntArray? {
        val g1 = 4; val g2 = 12
        val hist = Array(3) { IntArray(256) }; var total = 0
        val ox0 = max(0, bx - g2); val oy0 = max(0, by - g2); val ox1 = min(pw, bx + bw + g2); val oy1 = min(ph, by + bh + g2)
        for (y in oy0 until oy1) for (x in ox0 until ox1) {
            val inInner = x >= bx - g1 && x < bx + bw + g1 && y >= by - g1 && y < by + bh + g1
            if (inInner) continue
            val p = px[y * pw + x]; hist[0][(p shr 16) and 255]++; hist[1][(p shr 8) and 255]++; hist[2][p and 255]++; total++
        }
        val full = (bw + 2 * g2) * (bh + 2 * g2) - (bw + 2 * g1) * (bh + 2 * g1) // 고리 픽셀 수(가장자리에 잘리지 않았을 때)
        if (total < 400 || total < 0.2f * full) return null // 상자가 이미지 모서리에 붙어 있으면 둘레의 일부만 있어도 됨
        val med = IntArray(3)
        for (c in 0..2) {
            var acc = 0; var mi = 0
            while (mi < 255 && acc + hist[c][mi] < total / 2) { acc += hist[c][mi]; mi++ }
            med[c] = mi
            var near = 0; for (v in max(0, mi - 10)..min(255, mi + 10)) near += hist[c][v]
            if (near < 0.90f * total) return null
        }
        return med
    }

    /**
     * 검출 결과와 풀 이미지들의 상자 조각으로 모델을 만든다.
     * 상자 크기 bw x bh, regs 는 모두 같은 크기(rw x rh)이고 상자는 조각 안 (padL, padT)에서 시작(이미지 가장자리에서는 여백이 줄어듦)
     */
    fun buildModel(det: WmDet, s: Int, w: Int, tw: Int, bw: Int, bh: Int, regs: List<WmReg>, rw: Int, rh: Int, padL: Int, padT: Int): WmModel? {
        if (regs.size < 3) return null
        if (rw < bw + padL || rh < bh + padT) return null
        // 윤곽의 중앙값 지도 → 풋프린트(워터마크가 차지하는 자리)
        val gs = regs.map { r -> colorGradMag(r.px, rw, rh) }
        val med = FloatArray(rw * rh); val tmp = FloatArray(gs.size)
        for (i in med.indices) { for (k in gs.indices) tmp[k] = gs[k][i]; tmp.sort(); med[i] = tmp[tmp.size / 2] }
        val sorted = med.copyOf(); sorted.sort()
        val p95 = sorted[(sorted.size * 0.95f).toInt().coerceIn(0, sorted.size - 1)]
        val thr = max(6f, 0.2f * p95)
        var fp = BooleanArray(rw * rh) { med[it] > thr }
        fp = erode(dilate(fp, rw, rh, 3), rw, rh, 3)
        fp = fillHoles(fp, rw, rh)
        fp = dilate(fp, rw, rh, 2)
        val support = BooleanArray(bw * bh)
        for (y in 0 until bh) for (x in 0 until bw) support[y * bw + x] = fp[(y + padT) * rw + x + padL]
        val nyF = det.tyF * s
        // 확인용 풋프린트(축소 칸): 템플릿 윤곽이 있는 자리를 2칸 팽창
        val tl = FloatArray(det.tgx.size) { det.tgx[it] + det.tgy[it] }; val tsorted = tl.copyOf(); tsorted.sort()
        val tthr = max(6f, 0.2f * tsorted[(tsorted.size * 0.95f).toInt().coerceIn(0, tsorted.size - 1)])
        val tsup = dilate(BooleanArray(tl.size) { tl[it] > tthr }, det.tbw, det.tbh, 2)
        val m = WmModel(det.frame, w, s, tw, det.tx0 * s, nyF, bw, bh, det.tx0, det.tyF, det.tbw, det.tbh, det.tgx, det.tgy, tsup, support, det.tref)
        for (r in regs) {
            val j = ringFlat(r.px, rw, rh, padL, padT, bw, bh) ?: continue
            var ax = 0; var ay = 0
            if (m.n >= 1 && det.frame != TL && det.frame != BL) { val al = m.align(r.px, rw, rh, padL, padT, 2); ax = al[0]; ay = al[1] }
            val region = IntArray(bw * bh)
            for (y in 0 until bh) System.arraycopy(r.px, (y + padT + ay) * rw + padL + ax, region, y * bw, bw)
            if (!m.interiorFlat(region, j)) continue
            m.addFlat(region, j)
        }
        m.solve()
        return m
    }

    private fun fillHoles(b: BooleanArray, w: Int, h: Int): BooleanArray {
        val outside = BooleanArray(w * h); val stack = IntArray(w * h); var sp = 0
        fun push(i: Int) { if (!b[i] && !outside[i]) { outside[i] = true; stack[sp++] = i } }
        for (x in 0 until w) { push(x); push((h - 1) * w + x) }
        for (y in 0 until h) { push(y * w); push(y * w + w - 1) }
        while (sp > 0) { val p = stack[--sp]; val x = p % w; val y = p / w
            if (x > 0) push(p - 1); if (x < w - 1) push(p + 1); if (y > 0) push(p - w); if (y < h - 1) push(p + w) }
        return BooleanArray(w * h) { !outside[it] }
    }
}

/**
 * 워터마크 모델: 위치(프레임 기준) + 윤곽 템플릿 + 지워질 자리(support) + 투명도/색 학습(회귀 합계).
 * 상자 좌표: 가로는 왼쪽 기준 nx, 세로는 TL이면 위에서 nyF, BL이면 이미지 아래에서 nyF 떨어진 상자 아랫변.
 */
/** 워터마크 제거용 작은 신경망(안드로이드: ONNX Runtime, PC 시험: 같은 모델). 입력 x = [9,h,w](RGB 0..1, 채널별 단색 배경 변화량 d/255 ×3, 그 배경 색/255 ×3), 출력 [3,h,w] 0..1. h,w 는 8의 배수 */
interface WmNet { fun infer(x: FloatArray, h: Int, w: Int): FloatArray? }

class WmModel(
    val frame: Int, val w: Int, val s: Int, val tw: Int,
    val nx: Int, val nyF: Int, val nbw: Int, val nbh: Int,
    val tx0: Int, val tyF: Int, val tbw: Int, val tbh: Int,
    val tgx: FloatArray, val tgy: FloatArray, val tsup: BooleanArray, val support: BooleanArray, val tref: Int
) {
    val size = nbw * nbh
    var n = 0
    val sx = DoubleArray(3); val sxx = DoubleArray(3)
    val sy = Array(3) { FloatArray(size) }; val sxy = Array(3) { FloatArray(size) }
    @Volatile var u: FloatArray? = null
    @Volatile var v: Array<FloatArray>? = null
    @Volatile var diversity = 0.0   // 배경 색 다양성(제곱합) — LEARN_D 에 도달하면 학습 완료
    var cleaned = 0
    @Volatile var badStreak = 0     // 연속으로 '지웠는데 윤곽이 안 줄어' 취소된 횟수(워터마크 모양이 바뀌었다는 신호, 저장 안 함)

    fun nativeY(h: Int) = WmCore.nativeYOf(frame, nyF, nbh, h, tref, s)

    // 원본 해상도 윤곽 템플릿(평균 '관측-배경' 의 윤곽 상위 점들): 이미지마다 1~2px 어긋나는 위치를 맞추는 데 씀
    private var tnIdx: IntArray? = null
    private var tnVal: FloatArray? = null
    private var tnN = -1

    @Synchronized private fun ensureTemplate() {
        if (tnN == n && tnIdx != null) return
        tnN = n; tnIdx = null; tnVal = null
        if (n < 1) return
        val d = FloatArray(size) { i -> var a = 0f; for (c in 0..2) a += sy[c][i] / n - (sx[c] / n).toFloat(); a / 3f }
        val g = FloatArray(size)
        for (y in 1 until nbh - 1) { val r = y * nbw; for (x in 1 until nbw - 1) g[r + x] = abs(d[r + x + 1] - d[r + x - 1]) + abs(d[r + x + nbw] - d[r + x - nbw]) }
        val cand = ArrayList<Int>()
        for (i in 0 until size) if (support[i] && g[i] > 0f) cand.add(i)
        if (cand.size < 200) return
        cand.sortByDescending { g[it] }
        val k = min(cand.size, 6000)
        tnIdx = IntArray(k) { cand[it] }; tnVal = FloatArray(k) { g[cand[it]] }
    }

    /**
     * 상자 주변 조각(patch: pw x ph, 상자 시작이 (ox,oy), 둘레 여백 ≥ m)에서 모델 윤곽과 가장 잘 맞는 어긋남(dx,dy ∈ −m..m)을 찾음.
     * 위/아래 기준 프레임은 어긋남이 없으므로 호출하지 않아도 됨. 개선이 뚜렷하지 않으면 (0,0)
     */
    @Synchronized fun align(patch: IntArray, pw: Int, ph: Int, ox: Int, oy: Int, m: Int): IntArray {
        ensureTemplate()
        val idx = tnIdx ?: return intArrayOf(0, 0); val tv = tnVal ?: return intArrayOf(0, 0)
        val gr = FloatArray(pw * ph) { WmCore.gray(patch[it]) }
        val gp = FloatArray(pw * ph)
        for (y in 1 until ph - 1) { val r = y * pw; for (x in 1 until pw - 1) gp[r + x] = abs(gr[r + x + 1] - gr[r + x - 1]) + abs(gr[r + x + pw] - gr[r + x - pw]) }
        val cnt = idx.size
        var sb = 0.0; var sbb = 0.0; for (v in tv) { sb += v; sbb += v.toDouble() * v }
        val vb = sbb - sb * sb / cnt; if (vb < 1e-6) return intArrayOf(0, 0)
        var best = -2.0; var bdx = 0; var bdy = 0; var c0 = -2.0
        for (dy in -m..m) for (dx in -m..m) {
            var sa = 0.0; var saa = 0.0; var sab = 0.0; var ok = true
            for (k in 0 until cnt) {
                val i = idx[k]; val px = ox + dx + i % nbw; val py = oy + dy + i / nbw
                if (px < 1 || py < 1 || px >= pw - 1 || py >= ph - 1) { ok = false; break }
                val a = gp[py * pw + px].toDouble(); sa += a; saa += a * a; sab += a * tv[k]
            }
            if (!ok) continue
            val va = saa - sa * sa / cnt; if (va < 1e-6) continue
            val c = (sab - sa * sb / cnt) / sqrt(va * vb)
            if (dx == 0 && dy == 0) c0 = c
            if (c > best) { best = c; bdx = dx; bdy = dy }
        }
        return if (best > c0 + 0.03) intArrayOf(bdx, bdy) else intArrayOf(0, 0)
    }

    private var hpN = -1; private var hpArr: FloatArray? = null
    /** 단색 배경 표본들의 평균 '어두워진 정도'(회색) 중 윤곽 성분: d - (반지름 8px 평균) */
    @Synchronized fun hpTemplate(): FloatArray? {
        if (n < 1) return null
        if (hpN == n && hpArr != null) return hpArr
        val d = FloatArray(size) { i -> var a = 0f; for (c in 0..2) a += (sx[c] / n).toFloat() - sy[c][i] / n; a / 3f }
        val r = 8
        val t = FloatArray(size); val bl = FloatArray(size)
        for (y in 0 until nbh) { var acc = 0f; var cnt = 0; val row = y * nbw
            for (x in 0 until min(r, nbw)) { acc += d[row + x]; cnt++ }
            for (x in 0 until nbw) { val add = x + r; if (add < nbw) { acc += d[row + add]; cnt++ }; val rem = x - r - 1; if (rem >= 0) { acc -= d[row + rem]; cnt-- }; t[row + x] = acc / cnt } }
        for (x in 0 until nbw) { var acc = 0f; var cnt = 0
            for (y in 0 until min(r, nbh)) { acc += t[y * nbw + x]; cnt++ }
            for (y in 0 until nbh) { val add = y + r; if (add < nbh) { acc += t[add * nbw + x]; cnt++ }; val rem = y - r - 1; if (rem >= 0) { acc -= t[rem * nbw + x]; cnt-- }; bl[y * nbw + x] = acc / cnt } }
        val hp = FloatArray(size) { d[it] - bl[it] }
        hpArr = hp; hpN = n
        return hp
    }

    /** 단색 배경(j) 위의 표본 추가: region = 상자 크기 픽셀 */
    @Synchronized fun addFlat(region: IntArray, j: IntArray) {
        if (n >= 60) return
        for (i in 0 until size) {
            val p = region[i]
            val r = (p shr 16) and 255; val g = (p shr 8) and 255; val b = p and 255
            sy[0][i] = sy[0][i] + r; sy[1][i] = sy[1][i] + g; sy[2][i] = sy[2][i] + b
            sxy[0][i] = sxy[0][i] + j[0] * r; sxy[1][i] = sxy[1][i] + j[1] * g; sxy[2][i] = sxy[2][i] + j[2] * b
            if (abs(r - j[0]) > 12 || abs(g - j[1]) > 12 || abs(b - j[2]) > 12) support[i] = true // 실제로 달라진 자리는 워터마크 자리
        }
        n++
        for (c in 0..2) { sx[c] = sx[c] + j[c]; sxx[c] = sxx[c] + j[c].toDouble() * j[c] }
    }

    /** 회귀: 픽셀마다 관측값 I = 배경 J × (1-a) + v. 서로 다른 배경색 표본이 충분할 때만 a, v 를 구함 */
    @Synchronized fun solve(): Boolean {
        if (n < 1) return false
        var d0 = 0.0; for (c in 0..2) d0 += sxx[c] - sx[c] * sx[c] / n
        val useArt = false
        if (false && d0 < WmCore.LEARN_D && aN >= 8 && mCnt > 0) return solveApprox() // 근사 학습은 시험에서 이득이 없어 끔(수동 보정 사용)
        val wt = 0.0
        val nn = n + wt
        if (nn < 2.0) { diversity = d0; return false }
        val mc = DoubleArray(3) { if (mCnt > 0) mSum[it] / mCnt else 0.0 }
        val sx2 = DoubleArray(3) { sx[it] + wt * mc[it] }; val sxx2 = DoubleArray(3) { sxx[it] + wt * mc[it] * mc[it] }
        var d = 0.0; for (c in 0..2) d += sxx2[c] - sx2[c] * sx2[c] / nn
        diversity = d
        if (d < WmCore.LEARN_D) return false
        val un = FloatArray(size); val vn = Array(3) { FloatArray(size) }
        for (i in 0 until size) {
            if (!support[i]) continue
            var num = 0.0; val sy2 = DoubleArray(3)
            for (c in 0..2) {
                val abar = if (useArt) aSum[c][i].toDouble() / aN else 0.0
                sy2[c] = sy[c][i] + wt * abar
                num += sxy[c][i] + wt * mc[c] * abar - sx2[c] * sy2[c] / nn
            }
            val slope = (num / d).toFloat().coerceIn(0.03f, 1.2f)
            val a = 1f - slope
            if (a < 0.02f) continue
            un[i] = min(a, 0.97f)
            for (c in 0..2) vn[c][i] = (sy2[c] / nn - slope * sx2[c] / nn).toFloat()
        }
        v = vn; u = un; approx = useArt
        return true
    }

    val learned get() = u != null

    /**
     * 근사 학습: 단색 배경 표본에서 d(p)=어두워진 정도(워터마크 모양), 그림 위 표본의 평균 Ā(p)와 그림 평균색 m 으로 로고의 색 L 하나를 구함.
     *  평균적으로 Ā = m - u(m-L), u = d/(Jf-L)  ⇒  (m-Ā)/d = (m-L)/(Jf-L) = ρ  (전 픽셀에 걸쳐 최소제곱 → 잡음이 작음)
     * 이후 u = d/(Jf-L), v = u·L 로 J=(I-v)/(1-u) 복원
     */
    private fun solveApprox(): Boolean {
        var jf = 0.0; for (c in 0..2) jf += sx[c] / n; jf /= 3.0
        var m = 0.0; for (c in 0..2) m += mSum[c] / mCnt; m /= 3.0
        var num = 0.0; var den = 0.0
        val dArr = FloatArray(size)
        for (i in 0 until size) {
            var d = 0.0; var ab = 0.0
            for (c in 0..2) { d += sx[c] / n - sy[c][i] / n; ab += aSum[c][i].toDouble() / aN }
            d /= 3.0; ab /= 3.0
            dArr[i] = d.toFloat()
            if (!support[i] || d < 8.0) continue
            num += d * (m - ab); den += d * d
        }
        diversity = den
        if (den < 1.0) return false
        val rho = num / den
        if (rho < 0.02 || rho > 0.98 || jf - m < 20.0) return false
        val l = ((m - jf * rho) / (1.0 - rho)).coerceIn(0.0, 255.0)
        val un = FloatArray(size); val vn = Array(3) { FloatArray(size) }
        for (i in 0 until size) {
            if (!support[i] || dArr[i] < 4f) continue
            val uu = (dArr[i] / (jf - l)).toFloat().coerceIn(0f, 0.9f)
            un[i] = uu; for (c in 0..2) vn[c][i] = (uu * l).toFloat()
        }
        u = un; v = vn; approx = true; estL = l.toFloat()
        return true
    }
    var estL = -1f
    @Volatile var lastL = -1f; @Volatile var lastG = -1f

    /** 상자 테두리(안쪽 6px 띠)도 같은 단색이면 true: 상자 안이 정말 단색 배경인지 확인(그림이 걸쳐 있으면 학습 표본으로 쓰지 않음). 워터마크는 상자 가장자리에 닿지 않음 */
    fun interiorFlat(region: IntArray, j: IntArray): Boolean {
        val b = 6; var cnt = 0; var near = 0
        for (y in 0 until nbh) for (x in 0 until nbw) {
            if (x >= b && x < nbw - b && y >= b && y < nbh - b) continue
            cnt++
            val p = region[y * nbw + x]
            if (abs(((p shr 16) and 255) - j[0]) <= 10 && abs(((p shr 8) and 255) - j[1]) <= 10 && abs((p and 255) - j[2]) <= 10) near++
        }
        return cnt < 200 || near >= 0.85f * cnt
    }

    // 그림 위(단색 아님)에서 본 표본의 평균: 배경이 다양한 그림들의 평균색(m)을 '가짜 단색 표본'으로 써서, 흰 배경 표본과 함께 투명도를 근사
    var aN = 0
    val aSum = Array(3) { FloatArray(size) }
    val mSum = DoubleArray(3); var mCnt = 0
    var approx = false

    /** 워터마크 밖(support 밖) 상자 픽셀의 평균색을 m 으로, 전체 픽셀은 평균 관측값으로 누적 */
    @Synchronized fun addArt(region: IntArray) {
        if (aN >= 80) return
        var cnt = 0; var r = 0.0; var g = 0.0; var b = 0.0
        for (i in 0 until size) { if (support[i]) continue; val p = region[i]; r += (p shr 16) and 255; g += (p shr 8) and 255; b += p and 255; cnt++ }
        if (cnt < 100) return
        mSum[0] += r / cnt; mSum[1] += g / cnt; mSum[2] += b / cnt; mCnt++
        for (i in 0 until size) { val p = region[i]; aSum[0][i] = aSum[0][i] + ((p shr 16) and 255); aSum[1][i] = aSum[1][i] + ((p shr 8) and 255); aSum[2][i] = aSum[2][i] + (p and 255) }
        aN++
    }

    /** 저장용 사본(스레드 안전하게 배열을 복사) */
    @Synchronized fun copy(): WmModel {
        val c = WmModel(frame, w, s, tw, nx, nyF, nbw, nbh, tx0, tyF, tbw, tbh, tgx.copyOf(), tgy.copyOf(), tsup.copyOf(), support.copyOf(), tref)
        c.n = n
        for (k in 0..2) { c.sx[k] = sx[k]; c.sxx[k] = sxx[k]; System.arraycopy(sy[k], 0, c.sy[k], 0, size); System.arraycopy(sxy[k], 0, c.sxy[k], 0, size) }
        c.cleaned = cleaned
        c.aN = aN; c.mCnt = mCnt; for (k in 0..2) { c.mSum[k] = mSum[k]; System.arraycopy(aSum[k], 0, c.aSum[k], 0, size) }
        return c
    }

    /**
     * 상자 조각(region)에서 워터마크 제거. 학습됐으면 J=(I-v)/(1-a)로 되돌리고,
     * 아니고 flat(둘레 단색)이 있으면 그 색으로 채움. 바뀐 게 있으면 true
     */
    /** 지우기 + (그림 위일 때) 워터마크 자리를 살짝 흐리게 해 남은 흔적이 덜 보이게 함 */
    fun clean(region: IntArray, flat: IntArray?, gain: Float = 0f, logoL: Float = 140f, nn: WmNet? = null): Boolean {
        // 그림 위(단색 아님)이고 정확 학습이 안 됐으면, 계산식 결과와 신경망 결과 중 워터마크 윤곽이 더 적게 남는 쪽을 씀
        val orig = if (nn != null && flat == null && u == null && n >= 1) region.copyOf() else null
        val c = cleanCore(region, flat, gain, logoL)
        if (orig != null && nn != null) {
            try {
                val cand = nnClean(orig, nn)
                if (cand != null) {
                    val pa = abs(WmCore.regionPresence(this, if (c) region else orig)); val pn = abs(WmCore.regionPresence(this, cand))
                    if (pn < pa) { System.arraycopy(cand, 0, region, 0, size); usedNn = true; smudge(region); return true } else usedNn = false
                }
            } catch (_: Throwable) { usedNn = false }
        }
        if (flat == null) { smudge(region); return true }
        return c
    }
    @Volatile var usedNn = false

    /** 신경망으로 지운 결과(워터마크 자리 + 여백 안에서만 바꿈). 실패하면 null */
    private fun nnClean(orig: IntArray, nn: WmNet): IntArray? {
        var x0 = nbw; var y0 = nbh; var x1 = -1; var y1 = -1
        for (i in 0 until size) if (support[i]) { val x = i % nbw; val y = i / nbw; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y }
        if (x1 < 0) return null
        val m = 12
        x0 = max(0, x0 - m); y0 = max(0, y0 - m); x1 = min(nbw - 1, x1 + m); y1 = min(nbh - 1, y1 + m)
        val bw = x1 - x0 + 1; val bh = y1 - y0 + 1
        val W = (bw + 7) / 8 * 8; val H = (bh + 7) / 8 * 8
        if (W.toLong() * H > 700_000L) return null
        val hw = W * H; val x = FloatArray(9 * hw)
        val jc = FloatArray(3) { (sx[it] / n).toFloat() }
        for (yy in 0 until H) { val sy0 = min(nbh - 1, y0 + yy)
            for (xx in 0 until W) {
                val si = sy0 * nbw + min(nbw - 1, x0 + xx); val p = orig[si]; val o = yy * W + xx
                x[o] = ((p shr 16) and 255) / 255f; x[hw + o] = ((p shr 8) and 255) / 255f; x[2 * hw + o] = (p and 255) / 255f
                for (c in 0..2) { x[(3 + c) * hw + o] = (jc[c] - sy[c][si] / n) / 255f; x[(6 + c) * hw + o] = jc[c] / 255f }
            } }
        val y = nn.infer(x, H, W) ?: return null
        if (y.size < 3 * hw) return null
        // 워터마크 자리(지지대를 4px 넓힘) 안에서만 신경망 결과로 바꿈
        val dil = BooleanArray(size)
        val r = 4
        val tmp = BooleanArray(size)
        for (yy in 0 until nbh) for (xx in 0 until nbw) { if (!support[yy * nbw + xx]) continue; for (k in max(0, xx - r)..min(nbw - 1, xx + r)) tmp[yy * nbw + k] = true }
        for (yy in 0 until nbh) for (xx in 0 until nbw) { if (!tmp[yy * nbw + xx]) continue; for (k in max(0, yy - r)..min(nbh - 1, yy + r)) dil[k * nbw + xx] = true }
        val out = orig.copyOf()
        for (yy in 0 until bh) for (xx in 0 until bw) {
            val si = (y0 + yy) * nbw + x0 + xx; if (!dil[si]) continue
            val o = yy * W + xx
            val r2 = (y[o] * 255f + 0.5f).toInt().coerceIn(0, 255); val g2 = (y[hw + o] * 255f + 0.5f).toInt().coerceIn(0, 255); val b2 = (y[2 * hw + o] * 255f + 0.5f).toInt().coerceIn(0, 255)
            out[si] = (0xFF shl 24) or (r2 shl 16) or (g2 shl 8) or b2
        }
        return out
    }

    /**
     * 번지기(스머지): 워터마크가 진했던 자리(단색 배경에서 본 변화량이 큰 곳 + 2px)를 '워터마크가 아닌 주변 픽셀 색'으로만 채워 번지게 한 뒤,
     * 진했던 정도에 비례해 지운 결과와 섞음. 지운 뒤에도 남은 흔적·귀신 윤곽을 주변 색으로 덮어 없애는 용도
     */
    var smudgeAmt = 0.85f
    private fun smudge(region: IntArray) {
        if (n < 1) { softBlur(region); return }
        val sg = FloatArray(size)
        for (i in 0 until size) { var d = 0f; for (c in 0..2) d += (sx[c] / n).toFloat() - sy[c][i] / n; sg[i] = min(1f, abs(d / 3f) / 40f) }
        // 반지름 2 최대값 필터(얇은 윤곽까지 포함)
        val t1 = FloatArray(size); val st = FloatArray(size)
        for (y in 0 until nbh) for (x in 0 until nbw) { var m = 0f; for (k in max(0, x - 2)..min(nbw - 1, x + 2)) m = max(m, sg[y * nbw + k]); t1[y * nbw + x] = m }
        for (y in 0 until nbh) for (x in 0 until nbw) { var m = 0f; for (k in max(0, y - 2)..min(nbh - 1, y + 2)) m = max(m, t1[k * nbw + x]); st[y * nbw + x] = m }
        val mask = BooleanArray(size) { st[it] > 0.2f }
        var cnt0 = 0; for (b in mask) if (b) cnt0++
        if (cnt0 < 30 || cnt0 > size * 9 / 10) return
        val r = FloatArray(size); val g = FloatArray(size); val b = FloatArray(size)
        val known = BooleanArray(size)
        for (i in 0 until size) { val p = region[i]; r[i] = ((p shr 16) and 255).toFloat(); g[i] = ((p shr 8) and 255).toFloat(); b[i] = (p and 255).toFloat(); known[i] = !mask[i] }
        // 1) 바깥쪽부터 한 겹씩 이웃(이미 아는 픽셀)의 평균으로 채움
        var left = cnt0; var guard = 0
        val fillOrder = ArrayList<Int>()
        while (left > 0 && guard++ < 400) {
            val newly = ArrayList<Int>(); val nr = ArrayList<Float>(); val ng = ArrayList<Float>(); val nb = ArrayList<Float>()
            for (i in 0 until size) {
                if (known[i]) continue
                val x = i % nbw; val y = i / nbw; var sr = 0f; var sgg = 0f; var sb = 0f; var c = 0
                for (yy in max(0, y - 1)..min(nbh - 1, y + 1)) for (xx in max(0, x - 1)..min(nbw - 1, x + 1)) { val k = yy * nbw + xx; if (known[k]) { sr += r[k]; sgg += g[k]; sb += b[k]; c++ } }
                if (c > 0) { newly.add(i); nr.add(sr / c); ng.add(sgg / c); nb.add(sb / c) }
            }
            if (newly.isEmpty()) break
            for (j in newly.indices) { val i = newly[j]; r[i] = nr[j]; g[i] = ng[j]; b[i] = nb[j]; known[i] = true; fillOrder.add(i) }
            left -= newly.size
        }
        if (left > 0) return
        // 2) 채운 자리만 이웃과 매끄럽게(라플라스 완화)
        for (it in 0 until 30) for (i in fillOrder) {
            val x = i % nbw; val y = i / nbw; var sr = 0f; var sgg = 0f; var sb = 0f; var c = 0
            if (x > 0) { sr += r[i - 1]; sgg += g[i - 1]; sb += b[i - 1]; c++ }
            if (x < nbw - 1) { sr += r[i + 1]; sgg += g[i + 1]; sb += b[i + 1]; c++ }
            if (y > 0) { sr += r[i - nbw]; sgg += g[i - nbw]; sb += b[i - nbw]; c++ }
            if (y < nbh - 1) { sr += r[i + nbw]; sgg += g[i + nbw]; sb += b[i + nbw]; c++ }
            if (c > 0) { r[i] = sr / c; g[i] = sgg / c; b[i] = sb / c }
        }
        // 3) 진했던 정도에 비례해 섞음
        for (i in fillOrder) {
            val a = (min(1f, 0.55f + st[i]) * smudgeAmt * (if (u != null) 0.5f else 1f)).coerceIn(0f, 1f); val p = region[i]
            val nr = (((p shr 16) and 255) * (1f - a) + r[i] * a).toInt().coerceIn(0, 255)
            val ng = (((p shr 8) and 255) * (1f - a) + g[i] * a).toInt().coerceIn(0, 255)
            val nb = ((p and 255) * (1f - a) + b[i] * a).toInt().coerceIn(0, 255)
            region[i] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
        }
    }

    private fun softBlur(region: IntArray) {
        val src = region.copyOf(); val r = 2
        for (i in 0 until size) {
            if (!support[i]) continue
            val x = i % nbw; val y = i / nbw
            var sr = 0; var sg = 0; var sb = 0; var cnt = 0
            for (yy in max(0, y - r)..min(nbh - 1, y + r)) for (xx in max(0, x - r)..min(nbw - 1, x + r)) { val p = src[yy * nbw + xx]; sr += (p shr 16) and 255; sg += (p shr 8) and 255; sb += p and 255; cnt++ }
            val p0 = src[i]
            val nr = (((p0 shr 16) and 255) * 0.25f + (sr / cnt) * 0.75f).toInt(); val ng = (((p0 shr 8) and 255) * 0.25f + (sg / cnt) * 0.75f).toInt(); val nb = ((p0 and 255) * 0.25f + (sb / cnt) * 0.75f).toInt()
            region[i] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
        }
    }

    private fun cleanCore(region: IntArray, flat: IntArray?, gain: Float, logoL: Float): Boolean {
        val un = u; val vn = v
        var changed = false
        // 단색 배경 표본에서 본 워터마크 모양 d 로 u = g·d/(배경-L), J=(I-u·L)/(1-u) 를 만들되, 로고 색 L 과 세기 g 는
        // 후보들 중 '워터마크 자리의 윤곽 에너지'가 가장 작아지는 값을 이미지마다 자동으로 고름(사람 눈이 하는 것과 같은 기준: 가장 매끈해지는 쪽)
        if (flat == null && gain > 0f && un == null && n >= 1) {
            val jf = ((sx[0] + sx[1] + sx[2]) / (3.0 * n)).toFloat()
            if (jf < 100f) return false
            val dg = FloatArray(size) { i -> var d = 0f; for (c in 0..2) d += (sx[c] / n).toFloat() - sy[c][i] / n; d / 3f }
            val gI = FloatArray(size) { WmCore.gray(region[it]) }
            val idx = ArrayList<Int>(); for (i in 0 until size) if (support[i] && dg[i] >= 4f) idx.add(i)
            if (idx.size < 300) return false
            val ia = idx.toIntArray()
            // 후보(균일 불투명도 ū, 또는 로고색 L·세기 g)를 적용해 보고, 워터마크 윤곽 모양과의 상관(regionPresence)이 가장 낮아지는 값을 고름
            // (그림 윤곽이 강한 이미지에선 '전체 윤곽 에너지'가 거의 안 변해서 어두운 그림을 못 지웠기 때문에, 워터마크 모양에 특화된 지표를 씀)
            val tmp = IntArray(size)
            val pr = FloatArray(ia.size); val pg = FloatArray(ia.size); val pb = FloatArray(ia.size)
            for (k in ia.indices) { val p = region[ia[k]]; pr[k] = ((p shr 16) and 255).toFloat(); pg[k] = ((p shr 8) and 255).toFloat(); pb[k] = (p and 255).toFloat() }
            // mode 0: 로고색 L·세기 g,  mode 1: 균일 불투명도 ū (로고색은 배경-d/ū 로 자리마다 다름)
            fun jOf(i: Int, mode: Int, a: Float, g: Float, v: Float): Float =
                if (mode == 1) (v + dg[i] - a * jf) / (1f - a)
                else { val uu = (g * dg[i] / (jf - a)).coerceIn(0f, 0.9f); (v - uu * a) / (1f - uu) }
            fun applyTo(dst: IntArray, mode: Int, a: Float, g: Float) {
                for (k in ia.indices) { val i = ia[k]
                    val r = jOf(i, mode, a, g, pr[k]).toInt().coerceIn(0, 255); val g2 = jOf(i, mode, a, g, pg[k]).toInt().coerceIn(0, 255); val b2 = jOf(i, mode, a, g, pb[k]).toInt().coerceIn(0, 255)
                    dst[i] = (0xFF shl 24) or (r shl 16) or (g2 shl 8) or b2 }
            }
            fun score(mode: Int, a: Float, g: Float): Float { System.arraycopy(region, 0, tmp, 0, size); applyTo(tmp, mode, a, g); return abs(WmCore.regionPresence(this, tmp)) }
            val p0 = abs(WmCore.regionPresence(this, region))
            var best = p0; var bMode = -1; var bA = 0f; var bG = 0f
            var uq = 0.25f
            while (uq <= 0.901f) { val e = score(1, uq, 1f); if (e < best) { best = e; bMode = 1; bA = uq; bG = 1f }; uq += 0.025f }
            // 두 모델을 모두 시험하되, 자유도가 많은 쪽(로고색 L)은 확실히 더 낮을 때만 채택 — 지표에 과적합해서 엉뚱한 값을 고르기 쉬움
            val bestU = best; val aU = bA
            var bestF = p0; var aF = 0f; var gF = 0f
            for (l in 20..230 step 30) for (g in floatArrayOf(0.5f, 0.8f, 1.1f, 1.4f, 1.7f)) {
                if (jf - l < 25f) continue
                val e = score(0, l.toFloat(), g * gain); if (e < bestF) { bestF = e; aF = l.toFloat(); gF = g * gain }
            }
            if (bestF < p0) {
                val l0 = aF; val g0 = gF
                for (dl in -20..20 step 10) for (dg2 in floatArrayOf(-0.15f, 0f, 0.15f)) {
                    val l = (l0 + dl).coerceIn(0f, 230f); val g = g0 + dg2
                    if (g <= 0.2f || jf - l < 25f) continue
                    val e = score(0, l, g); if (e < bestF) { bestF = e; aF = l; gF = g }
                }
            }
            if (bestF < 0.7f * bestU) { best = bestF; bMode = 0; bA = aF; bG = gF } else { best = bestU; bA = aU }
            if (bMode < 0 || best > 0.6f * p0) return false
            lastL = if (bMode == 1) -2f else bA; lastG = if (bMode == 1) bA else bG
            applyTo(region, bMode, bA, bG); changed = true
            return changed
        }
        if (un != null && vn != null && !(approx && flat != null)) { // 근사 학습이면 단색 배경 위는 정확한 채움을 우선
            val hole = BooleanArray(size)
            for (i in 0 until size) {
                val gs = if (approx && gain > 0f) gain else 1f
                val a = un[i] * gs; if (a < 0.02f) continue
                if (a > 0.9f) { hole[i] = true; continue }
                val inv = 1f / (1f - a); val p = region[i]
                val r = (((p shr 16) and 255) - vn[0][i] * gs) * inv; val g = (((p shr 8) and 255) - vn[1][i] * gs) * inv; val b = ((p and 255) - vn[2][i] * gs) * inv
                region[i] = (0xFF shl 24) or (r.toInt().coerceIn(0, 255) shl 16) or (g.toInt().coerceIn(0, 255) shl 8) or b.toInt().coerceIn(0, 255)
                changed = true
            }
            // 거의 불투명한 점은 주변 평균으로 메움
            var pass = 0; var left = hole.count { it }
            while (left > 0 && pass++ < 4) {
                val snap = region.copyOf(); val still = hole.copyOf()
                for (i in 0 until size) {
                    if (!hole[i]) continue
                    val x = i % nbw; val y = i / nbw; var sr = 0; var sg = 0; var sb = 0; var c = 0
                    for (dy in -2..2) for (dx in -2..2) { val xx = x + dx; val yy = y + dy; if (xx < 0 || yy < 0 || xx >= nbw || yy >= nbh) continue
                        val k = yy * nbw + xx; if (hole[k]) continue; val p = snap[k]; sr += (p shr 16) and 255; sg += (p shr 8) and 255; sb += p and 255; c++ }
                    if (c > 0) { region[i] = (0xFF shl 24) or ((sr / c) shl 16) or ((sg / c) shl 8) or (sb / c); still[i] = false; changed = true }
                }
                System.arraycopy(still, 0, hole, 0, size); left = hole.count { it }
            }
            return changed
        }
        if (flat != null) {
            val fc = (0xFF shl 24) or (flat[0] shl 16) or (flat[1] shl 8) or flat[2]
            for (i in 0 until size) if (support[i] && region[i] != fc) { region[i] = fc; changed = true }
        }
        return changed
    }

    // ---------------------------------------------------------------- 저장/불러오기
    fun write(o: DataOutputStream) {
        o.writeInt(4); o.writeInt(frame); o.writeInt(w); o.writeInt(s); o.writeInt(tw); o.writeInt(tref)
        o.writeInt(nx); o.writeInt(nyF); o.writeInt(nbw); o.writeInt(nbh)
        o.writeInt(tx0); o.writeInt(tyF); o.writeInt(tbw); o.writeInt(tbh)
        for (f in tgx) o.writeFloat(f)
        for (f in tgy) o.writeFloat(f)
        for (b in tsup) o.writeBoolean(b)
        for (b in support) o.writeBoolean(b)
        o.writeInt(n); for (c in 0..2) { o.writeDouble(sx[c]); o.writeDouble(sxx[c]) }
        for (c in 0..2) for (f in sy[c]) o.writeFloat(f)
        for (c in 0..2) for (f in sxy[c]) o.writeFloat(f)
        o.writeInt(cleaned)
        o.writeInt(aN); o.writeInt(mCnt); for (c in 0..2) o.writeDouble(mSum[c]); for (c in 0..2) for (f in aSum[c]) o.writeFloat(f)
    }

    companion object {
        fun read(i: DataInputStream): WmModel? {
            val ver = i.readInt(); if (ver != 3 && ver != 4) return null
            val frame = i.readInt(); val w = i.readInt(); val s = i.readInt(); val tw = i.readInt(); val tref = i.readInt()
            val nx = i.readInt(); val nyF = i.readInt(); val nbw = i.readInt(); val nbh = i.readInt()
            val tx0 = i.readInt(); val tyF = i.readInt(); val tbw = i.readInt(); val tbh = i.readInt()
            if (nbw <= 0 || nbh <= 0 || nbw * nbh > 4_000_000 || tbw * tbh > 1_000_000) return null
            val tgx = FloatArray(tbw * tbh) { i.readFloat() }
            val tgy = FloatArray(tbw * tbh) { i.readFloat() }
            val tsup = BooleanArray(tbw * tbh) { i.readBoolean() }
            val sup = BooleanArray(nbw * nbh) { i.readBoolean() }
            val m = WmModel(frame, w, s, tw, nx, nyF, nbw, nbh, tx0, tyF, tbw, tbh, tgx, tgy, tsup, sup, tref)
            m.n = i.readInt(); for (c in 0..2) { m.sx[c] = i.readDouble(); m.sxx[c] = i.readDouble() }
            for (c in 0..2) for (k in 0 until m.size) m.sy[c][k] = i.readFloat()
            for (c in 0..2) for (k in 0 until m.size) m.sxy[c][k] = i.readFloat()
            m.cleaned = i.readInt()
            if (ver >= 4) { m.aN = i.readInt(); m.mCnt = i.readInt(); for (c in 0..2) m.mSum[c] = i.readDouble(); for (c in 0..2) for (k in 0 until m.size) m.aSum[c][k] = i.readFloat() }
            m.solve()
            return m
        }
    }
}
