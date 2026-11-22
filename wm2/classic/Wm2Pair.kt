package com.example.comicviewer.wm2

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** 2차원 FFT(2의 거듭제곱 크기). 윤곽 방향장의 상관을 구하는 데 쓴다. */
internal object Fft {
    fun pow2(n: Int): Int { var p = 1; while (p < n) p = p shl 1; return p }

    private fun fft1(re: DoubleArray, im: DoubleArray, off: Int, stride: Int, n: Int, inv: Boolean, tr: DoubleArray, ti: DoubleArray) {
        for (i in 0 until n) { tr[i] = re[off + i * stride]; ti[i] = im[off + i * stride] }
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) { val a = tr[i]; tr[i] = tr[j]; tr[j] = a; val b = ti[i]; ti[i] = ti[j]; ti[j] = b }
        }
        var len = 2
        while (len <= n) {
            val ang = 2.0 * Math.PI / len * (if (inv) 1 else -1); val wr = Math.cos(ang); val wi = Math.sin(ang); val half = len shr 1
            var i = 0
            while (i < n) {
                var cr = 1.0; var ci = 0.0
                for (k in 0 until half) {
                    val a = i + k; val b = a + half
                    val xr = tr[b] * cr - ti[b] * ci; val xi = tr[b] * ci + ti[b] * cr
                    tr[b] = tr[a] - xr; ti[b] = ti[a] - xi; tr[a] += xr; ti[a] += xi
                    val nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr
                }
                i += len
            }
            len = len shl 1
        }
        val sc = if (inv) 1.0 / n else 1.0
        for (i in 0 until n) { re[off + i * stride] = tr[i] * sc; im[off + i * stride] = ti[i] * sc }
    }

    fun fft2(re: DoubleArray, im: DoubleArray, w: Int, h: Int, inv: Boolean) {
        val m = max(w, h); val tr = DoubleArray(m); val ti = DoubleArray(m)
        for (y in 0 until h) fft1(re, im, y * w, 1, w, inv, tr, ti)
        for (x in 0 until w) fft1(re, im, x, w, h, inv, tr, ti)
    }
}

/** 장 한 쌍의 윤곽 방향장을 맞춰 보고(상관 봉우리 = 변위 후보), 여러 장이 같은 상자를 지지하는지 군집으로 판정한다.
 *  위치가 고정·비율·흔들림·세로만 제각각·완전 제각각이어도 같은 방식으로 처리된다(정렬을 미리 가정하지 않음). */
internal object Wm2Pair {
    class Spec(val w: Int, val h: Int, val pw: Int, val ph: Int, val xr: DoubleArray, val xi: DoubleArray, val yr: DoubleArray, val yi: DoubleArray)
    class Prop(val partner: Int, val dx: Int, val dy: Int, val x: Int, val y: Int, val w: Int, val h: Int, val score: Float, val dom: Float, val cnt: Int, val z: Float)
    class Found(val img: Int, val x: Int, val y: Int, val w: Int, val h: Int, val score: Float, val support: Int)
    /** 한 군집 = 같은 워터마크로 보이는 상자들(장마다 1개) + 특징량 */
    class Cluster(val ref: Int, val items: List<Found>, val support: Int, val score: Float, val meanZ: Float, val dom: Float, val dens: Float, val spread: Float, val nProps: Int, val k: Int)

    private fun spec(p: Plane, pw: Int, ph: Int): Spec {
        val xr = DoubleArray(pw * ph); val xi = DoubleArray(pw * ph); val yr = DoubleArray(pw * ph); val yi = DoubleArray(pw * ph)
        val mg = 3
        for (y in mg until p.h - mg) for (x in mg until p.w - mg) { val i = y * p.w + x; xr[y * pw + x] = p.vx[i].toDouble(); yr[y * pw + x] = p.vy[i].toDouble() }
        Fft.fft2(xr, xi, pw, ph, false); Fft.fft2(yr, yi, pw, ph, false)
        return Spec(p.w, p.h, pw, ph, xr, xi, yr, yi)
    }

    /** 변위 후보: (dx,dy)는 B의 점 = A의 점 + d 가 되는 이동량. z = 상관 표면에서의 표준화 점수 */
    private fun peaks(a: Spec, b: Spec, maxPeaks: Int): List<IntArray> {
        val n = a.pw * a.ph; val re = DoubleArray(n); val im = DoubleArray(n)
        for (i in 0 until n) {
            // conj(FA) * FB (x, y 채널 합)
            re[i] = a.xr[i] * b.xr[i] + a.xi[i] * b.xi[i] + a.yr[i] * b.yr[i] + a.yi[i] * b.yi[i]
            im[i] = a.xr[i] * b.xi[i] - a.xi[i] * b.xr[i] + a.yr[i] * b.yi[i] - a.yi[i] * b.yr[i]
        }
        Fft.fft2(re, im, a.pw, a.ph, true)
        // 겹치는 넓이가 충분한 변위만
        val maxDx = (a.w * 0.8).toInt(); val maxDy = (a.h * 0.8).toInt()
        // 강건한 표준화: 표본의 중앙값과 MAD (긴 선이 만드는 능선이 표준편차를 부풀리지 않게)
        val sample = DoubleArray(4001); var sc = 0
        val rnd = java.util.Random(7)
        while (sc < sample.size) { val iy = rnd.nextInt(a.ph); val ix = rnd.nextInt(a.pw); val dy = if (iy > a.ph / 2) iy - a.ph else iy; val dx = if (ix > a.pw / 2) ix - a.pw else ix
            if (abs(dy) > maxDy || abs(dx) > maxDx) continue; sample[sc++] = re[iy * a.pw + ix] }
        sample.sort(); val mean = sample[sample.size / 2]
        val dev = DoubleArray(sample.size) { abs(sample[it] - mean) }; dev.sort(); val sd = max(1e-9, dev[dev.size / 2] * 1.4826)
        val out = ArrayList<IntArray>(); val val_ = ArrayList<Double>()
        val rad = 3
        for (iy in 0 until a.ph) { val dy = if (iy > a.ph / 2) iy - a.ph else iy; if (abs(dy) > maxDy) continue
            for (ix in 0 until a.pw) { val dx = if (ix > a.pw / 2) ix - a.pw else ix; if (abs(dx) > maxDx) continue
                val v = re[iy * a.pw + ix]; val z = (v - mean) / sd
                if (z < Wm2Params.zMin) continue
                var isMax = true
                loop@ for (jy in -rad..rad) for (jx in -rad..rad) { if (jx == 0 && jy == 0) continue
                    val yy = (iy + jy + a.ph) % a.ph; val xx = (ix + jx + a.pw) % a.pw; if (re[yy * a.pw + xx] > v) { isMax = false; break@loop } }
                if (isMax) { out.add(intArrayOf(dx, dy, (z * 10).toInt())); val_.add(v) }
            } }
        val order = out.indices.sortedByDescending { val_[it] }.take(maxPeaks)
        return order.map { out[it] }
    }

    /** A와 B를 변위 d로 겹쳤을 때 윤곽 방향이 일치하는 덩어리(= 같은 구조) */
    private fun proposals(A: Plane, B: Plane, dx: Int, dy: Int, z: Int, partner: Int, s: Int): List<Prop> {
        val w = A.w; val h = A.h; val sm = FloatArray(w * h); val hit = BooleanArray(w * h); val rx = FloatArray(w * h); val ry = FloatArray(w * h)
        val mg = 4
        for (y in mg until h - mg) { val yb = y + dy; if (yb < mg || yb >= B.h - mg) continue
            for (x in mg until w - mg) { val xb = x + dx; if (xb < mg || xb >= B.w - mg) continue
                val ia = y * w + x; val ib = yb * B.w + xb
                val ax = A.vx[ia]; val ay = A.vy[ia]; val bx = B.vx[ib]; val by = B.vy[ib]
                val sx = ax + bx; val sy = ay + by; val r = sqrt(sx * sx + sy * sy) / 2f
                val na = sqrt(ax * ax + ay * ay); val nb = sqrt(bx * bx + by * by); val av = (na + nb) / 2f
                sm[ia] = r; rx[ia] = sx; ry[ia] = sy
                hit[ia] = av > 0f && r > Wm2Params.rMin && r / av > Wm2Params.qPair && A.mag[ia] > Wm2Params.lowMag && B.mag[ib] > Wm2Params.lowMag
            } }
        val out = ArrayList<Prop>()
        for (b in Wm2.blobs(Wm2.Cons(sm, hit, rx, ry), w, h, s)) if (b.why.isEmpty()) out.add(Prop(partner, dx, dy, b.x0, b.y0, b.w, b.h, b.sc / 100f, b.dom, b.cnt, z / 10f))
        return out
    }

    private fun iou(a: Prop, bx: Int, by: Int, bw: Int, bh: Int): Float {
        val ix = max(0, min(a.x + a.w, bx + bw) - max(a.x, bx)); val iy = max(0, min(a.y + a.h, by + bh) - max(a.y, by))
        val i = ix * iy; val u = a.w * a.h + bw * bh - i; return if (u > 0) i.toFloat() / u else 0f
    }

    /** 같은 너비의 장 묶음(planes: 축소 s)에서 군집 후보를 모두 돌려준다(채택은 호출한 쪽) */
    fun run(planes: List<Plane>, s: Int, log: MutableList<String>?): List<Cluster> {
        val k = planes.size
        val pw = Fft.pow2((planes.maxOf { it.w } * 1.25).toInt()); val ph = Fft.pow2((planes.maxOf { it.h } * 1.25).toInt())
        val specs = planes.map { spec(it, pw, ph) }
        val out = ArrayList<Cluster>()
        val refs = (0 until min(k, Wm2Params.refs))
        for (a in refs) {
            val props = ArrayList<Prop>()
            for (b in 0 until k) {
                if (b == a) continue
                val pk = peaks(specs[a], specs[b], Wm2Params.peaks)
                for (p in pk) props.addAll(proposals(planes[a], planes[b], p[0], p[1], p[2], b, s))
            }
            val sorted = props.sortedByDescending { it.score }
            class Cl(var x: Int, var y: Int, var w: Int, var h: Int) { val m = HashMap<Int, Prop>(); var n = 0 }
            val cls = ArrayList<Cl>()
            for (p in sorted) {
                val c = cls.firstOrNull { iou(p, it.x, it.y, it.w, it.h) >= Wm2Params.clIou }
                if (c == null) { val n = Cl(p.x, p.y, p.w, p.h); n.m[p.partner] = p; n.n = 1; cls.add(n) } else { c.n++; if (!c.m.containsKey(p.partner)) c.m[p.partner] = p }
            }
            for (c in cls) {
                if (c.m.size < 1) continue
                val mem = c.m.values.toList()
                val xs = mem.map { it.x }.sorted(); val ys = mem.map { it.y }.sorted()
                val x1s = mem.map { it.x + it.w }.sorted(); val y1s = mem.map { it.y + it.h }.sorted()
                val md = mem.size / 2
                val bx = xs[md]; val by = ys[md]; val bw = x1s[md] - bx; val bh = y1s[md] - by
                val sc = mem.sumOf { it.score.toDouble() }.toFloat()
                val spread = mem.map { 1f - iou(it, bx, by, bw, bh) }.average().toFloat()
                val items = ArrayList<Found>(); items.add(Found(a, bx, by, bw, bh, sc, mem.size + 1))
                for ((b, p) in c.m) items.add(Found(b, bx + p.dx, by + p.dy, bw, bh, sc, mem.size + 1))
                out.add(Cluster(a, items, mem.size + 1, sc, mem.map { it.z }.average().toFloat(), mem.map { it.dom }.average().toFloat(), mem.map { it.cnt.toFloat() / max(1, bw * bh) }.average().toFloat(), spread, c.n, k))
            }
        }
        return out
    }
}
