package com.example.comicviewer.wm2

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** wm2: 워터마크 인식 전용 엔진(순수 Kotlin, 안드로이드 의존 없음). 지우기와 무관. 기존 Wm* 코드와 독립. */
class Img(val w: Int, val h: Int, val px: IntArray)
class Mark(val x: Int, val y: Int, val w: Int, val h: Int, val kind: String, val score: Float)
internal class Plane(val w: Int, val h: Int, val mag: FloatArray, val vx: FloatArray, val vy: FloatArray)
class Wm2Result(val marks: List<List<Mark>>, val log: List<String>)

/** 시험에서 바꿔 볼 수 있는 값(-Dwm2.이름=값) */
object Wm2Params {
    private fun d(n: String, v: Double) = System.getProperty("wm2.$n")?.toDoubleOrNull() ?: v
    var scale = d("scale", 2.0).toInt()          // 분석용 축소 배율
    var quant = d("quant", 0.4)                   // 합의 분위(작을수록 '대부분의 장에 있어야' 함)
    var floorMag = d("floor", 5.0).toFloat()      // 합의 윤곽 세기 하한
    var relTex = d("rel", 2.0).toFloat()          // 주변 질감 대비 배수
    var dbg = d("dbg", 0.0).toInt()
    var mode = d("mode", 0.0).toInt()              // 0: 고정 정렬 합의만, 1: 쌍 변위 탐색만, 2: 둘 다
    var scalePair = d("scalepair", 4.0).toInt()
    var zMin = d("zmin", 4.0)                       // 상관 봉우리 표준화 점수 하한
    var qPair = d("qpair", 0.93).toFloat()          // 쌍 방향 일치도 하한
    var pairFrac = d("pairfrac", 0.5)               // 군집이 지지해야 하는 상대 장 비율
    var refs = d("refs", 3.0).toInt()               // 기준 장 수
    var peaks = d("peaks", 24.0).toInt()             // 쌍마다 보는 변위 후보 수
    var clIou = d("cliou", 0.4).toFloat()
    var minPair = d("minpair", 0.0).toFloat()
    var tileCell = d("tilecell", 0.015)          // 전면 무늬: 칸에서 반응 밀도 하한
    var tileFrac = d("tilefrac", 0.6)            // 반응 있는 칸 비율 하한
    var bandMin = d("bandmin", 24.0).toInt()     // 띠 두께(원본 px) 범위
    var bandMax = d("bandmax", 90.0).toInt()
    var bandInside = d("bandinside", 0.02)       // 띠 안쪽 반응 밀도 하한
    var dilate = d("dil", 3.0).toInt()
    var nullF = d("nullf", 2.0).toFloat()           // 실제 합의 세기가 귀무 지도의 몇 배 이상이어야 반응
    var nullBlur = d("nullblur", 4.0).toInt()
    var clip = d("clip", 25.0).toFloat()          // 윤곽 세기 상한(센 그림 선이 합의를 휘두르지 못하게)
    var qMin = d("q", 0.7).toFloat()               // 방향 합의도 하한(0~1)
    var rMin = d("r", 2.0).toFloat()               // 합의 세기 하한
    var cntMin = d("cnt", 0.6)                      // 윤곽이 있는 장 비율 하한
    var lowMag = d("lowmag", 2.0).toFloat()
    var domMax = d("dom", 0.85).toFloat()          // 윤곽 방향 쏠림 상한(넘으면 직선꼴)
    var nullK = d("nullk", 1.5).toFloat()           // 귀무 최고 점수의 몇 배 이상이어야 후보
    var relBest = d("relbest", 0.15).toFloat()      // 두 번째 이후 후보는 최고 점수의 이 비율 이상
    var minScore = d("minscore", 0.0).toFloat()
    var minPresent = d("minpresent", 0.6)
    var lineCover = d("linecover", 0.45)
    var maxW = d("maxw", 0.5)
    var maxHpx = d("maxhpx", 300.0)
}

object Wm2 {
    class Cand(val anchor: Double, val x: Int, val y: Int, val w: Int, val h: Int, val score: Float, val hits: Int, val present: BooleanArray, val offs: IntArray, val tw: Int, val th: Int, val why: String = "", val dom: Float = 0f, val meanR: Float = 0f, val kind: String = "mark")

    // ------------------------------------------------------------------ 준비
    internal fun prep(img: Img, s: Int): Plane {
        val w = img.w / s; val h = img.h / s
        val g = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var a = 0f
            for (dy in 0 until s) { var i = (y * s + dy) * img.w + x * s; for (dx in 0 until s) { val p = img.px[i++]; a += (p shr 16 and 255) * 299 + (p shr 8 and 255) * 587 + (p and 255) * 114 } }
            g[y * w + x] = a / (s * s * 1000f)
        }
        val b = FloatArray(w * h)
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val i = y * w + x
            b[i] = (g[i - w - 1] + 2 * g[i - w] + g[i - w + 1] + 2 * g[i - 1] + 4 * g[i] + 2 * g[i + 1] + g[i + w - 1] + 2 * g[i + w] + g[i + w + 1]) / 16f
        }
        val m = FloatArray(w * h); val ux = FloatArray(w * h); val uy = FloatArray(w * h); val cl = Wm2Params.clip
        for (y in 2 until h - 2) for (x in 2 until w - 2) {
            val i = y * w + x
            val gx = (b[i - w + 1] + 2 * b[i + 1] + b[i + w + 1] - b[i - w - 1] - 2 * b[i - 1] - b[i + w - 1]) / 8f
            val gy = (b[i + w - 1] + 2 * b[i + w] + b[i + w + 1] - b[i - w - 1] - 2 * b[i - w] - b[i - w + 1]) / 8f
            val mm = sqrt(gx * gx + gy * gy); m[i] = mm
            if (mm > 1e-3f) { val wt = min(mm, cl) / (mm * mm); ux[i] = wt * (gx * gx - gy * gy); uy[i] = wt * 2f * gx * gy }
        }
        return Plane(w, h, m, boxBlur(ux, w, h, 1), boxBlur(uy, w, h, 1))
    }

    internal fun boxBlur(a: FloatArray, w: Int, h: Int, r: Int): FloatArray {
        val t = FloatArray(w * h); val o = FloatArray(w * h); val pw = FloatArray(w + 1); val ph = FloatArray(h + 1)
        for (y in 0 until h) { val row = y * w; for (x in 0 until w) pw[x + 1] = pw[x] + a[row + x]
            for (x in 0 until w) { val x0 = max(0, x - r); val x1 = min(w - 1, x + r); t[row + x] = (pw[x1 + 1] - pw[x0]) / (x1 - x0 + 1) } }
        for (x in 0 until w) { for (y in 0 until h) ph[y + 1] = ph[y] + t[y * w + x]
            for (y in 0 until h) { val y0 = max(0, y - r); val y1 = min(h - 1, y + r); o[y * w + x] = (ph[y1 + 1] - ph[y0]) / (y1 - y0 + 1) } }
        return o
    }

    // ------------------------------------------------------------------ 합의
    internal class Cons(val sm: FloatArray, val hit: BooleanArray, val rx: FloatArray, val ry: FloatArray)

    /** 정렬된 장들의 윤곽 '방향'이 얼마나 일치하는가.
     *  null=true: 장마다 다른 크기로 순환 이동시켜 정렬만 깨뜨림(질감·직선의 방향 분포는 그대로) → '우연히 일치하는 수준' 측정 */
    private fun consensus(planes: List<Plane>, offs: IntArray, ws: Int, hmin: Int, nullSeed: Int, nullMap: FloatArray?): Cons {
        val k = planes.size; val sm = FloatArray(ws * hmin); val hit = BooleanArray(ws * hmin); val rxA = FloatArray(ws * hmin); val ryA = FloatArray(ws * hmin); val mg = 4
        val need = Math.ceil(Wm2Params.cntMin * k).toInt()
        val rnd = java.util.Random(1234L + nullSeed)
        val dxs = IntArray(k) { ws / 8 + rnd.nextInt(ws * 3 / 4) }; val dys = IntArray(k) { hmin / 8 + rnd.nextInt(hmin * 3 / 4) }
        val isNull = nullSeed != 0
        for (y in mg until hmin - mg) for (x in mg until ws - mg) {
            var sx = 0f; var sy = 0f; var sa = 0f; var cnt = 0
            for (i in 0 until k) {
                val p = planes[i]
                val j = if (isNull) ((y + dys[i]) % hmin + offs[i]) * ws + (x + dxs[i]) % ws else (y + offs[i]) * ws + x
                val vx = p.vx[j]; val vy = p.vy[j]
                sx += vx; sy += vy; sa += sqrt(vx * vx + vy * vy); if (p.mag[j] > Wm2Params.lowMag) cnt++
            }
            val r = sqrt(sx * sx + sy * sy) / k; val a = sa / k; val idx = y * ws + x
            sm[idx] = r; rxA[idx] = sx; ryA[idx] = sy
            val thr = if (nullMap != null) max(Wm2Params.rMin, Wm2Params.nullF * nullMap[idx]) else Wm2Params.rMin
            hit[idx] = r > thr && a > 0f && r / a > Wm2Params.qMin && cnt >= need
        }
        return Cons(sm, hit, rxA, ryA)
    }

    /** 귀무 지도: 두 번의 순환 이동으로 얻은 합의 세기의 최대를 주변 평균 */
    private fun nullMap(planes: List<Plane>, offs: IntArray, ws: Int, hmin: Int): FloatArray {
        val a = consensus(planes, offs, ws, hmin, 1, null).sm; val b = consensus(planes, offs, ws, hmin, 2, null).sm
        val m = FloatArray(ws * hmin) { max(a[it], b[it]) }
        return boxBlur(m, ws, hmin, Wm2Params.nullBlur)
    }

    /** 긴 직선(컷 테두리, 구분선) 제거: 한 줄/한 열의 대부분이 반응하면 그 줄을 지움 */
    internal fun killLines(hit: BooleanArray, ws: Int, hmin: Int, rowOut: BooleanArray? = null): IntArray {
        val rowKill = BooleanArray(hmin); val colKill = BooleanArray(ws); var kr = 0; var kc = 0
        for (y in 0 until hmin) { var c = 0; for (x in 0 until ws) if (hit[y * ws + x]) c++; if (c > Wm2Params.lineCover * ws) rowKill[y] = true }
        for (x in 0 until ws) { var c = 0; for (y in 0 until hmin) if (hit[y * ws + x]) c++; if (c > Wm2Params.lineCover * hmin * 0.5) colKill[x] = true }
        for (y in 0 until hmin) for (x in 0 until ws) {
            val kill = (rowKill[y] || (y > 0 && rowKill[y - 1]) || (y < hmin - 1 && rowKill[y + 1])) || (colKill[x] || (x > 0 && colKill[x - 1]) || (x < ws - 1 && colKill[x + 1]))
            if (kill) hit[y * ws + x] = false
        }
        for (y in 0 until hmin) if (rowKill[y]) kr++
        for (x in 0 until ws) if (colKill[x]) kc++
        if (rowOut != null) for (y in 0 until hmin) rowOut[y] = rowKill[y]
        return intArrayOf(kr, kc)
    }

    /** 윤곽 방향이 한 방향으로 쏠린 정도(0~1): 직선·테두리는 1에 가깝고 글자·도형은 낮다 */
    private fun dominance(c: Cons, lab: IntArray, id: Int, x0: Int, y0: Int, x1: Int, y1: Int, ws: Int): Float {
        val bins = IntArray(12); var n = 0
        for (y in y0..y1) for (x in x0..x1) { val i = y * ws + x; if (lab[i] == id && c.hit[i]) { var a = Math.atan2(c.ry[i].toDouble(), c.rx[i].toDouble()); if (a < 0) a += 2 * Math.PI; bins[min(11, (a / (2 * Math.PI) * 12).toInt())]++; n++ } }
        if (n == 0) return 1f
        var best = 0; for (b in 0 until 12) best = max(best, bins[b] + bins[(b + 1) % 12])
        return best.toFloat() / n
    }

    internal class Blob(val x0: Int, val y0: Int, val x1: Int, val y1: Int, val cnt: Int, val sc: Float, val dom: Float, val why: String, val lab: IntArray, val id: Int) {
        val w get() = x1 - x0 + 1
        val h get() = y1 - y0 + 1
    }

    /** 반응 지도에서 덩어리(팽창 후 연결)를 뽑아 모양 검사(작음·넓음·높음·직선꼴)까지 한다 */
    internal fun blobs(c: Cons, ws: Int, hh: Int, s: Int, killed: IntArray? = null, rowOut: BooleanArray? = null): List<Blob> {
        val hit = c.hit
        val kl = killLines(hit, ws, hh, rowOut); if (killed != null) { killed[0] = kl[0]; killed[1] = kl[1] }
        val r = Wm2Params.dilate
        val dil = BooleanArray(ws * hh)
        for (y in 0 until hh) for (x in 0 until ws) if (hit[y * ws + x]) for (dy in -r..r) { val yy = y + dy; if (yy < 0 || yy >= hh) continue; for (dx in -r..r) { val xx = x + dx; if (xx in 0 until ws) dil[yy * ws + xx] = true } }
        val lab = IntArray(ws * hh); val stack = IntArray(ws * hh); val out = ArrayList<Blob>()
        var nl = 0
        for (i0 in 0 until ws * hh) {
            if (!dil[i0] || lab[i0] != 0) continue
            nl++; var sp = 0; stack[sp++] = i0; lab[i0] = nl
            var x0 = Int.MAX_VALUE; var y0 = Int.MAX_VALUE; var x1 = -1; var y1 = -1; var cnt = 0; var sc = 0f
            while (sp > 0) {
                val i = stack[--sp]; val x = i % ws; val y = i / ws
                if (hit[i]) { cnt++; sc += c.sm[i]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y }
                if (x > 0 && dil[i - 1] && lab[i - 1] == 0) { lab[i - 1] = nl; stack[sp++] = i - 1 }
                if (x < ws - 1 && dil[i + 1] && lab[i + 1] == 0) { lab[i + 1] = nl; stack[sp++] = i + 1 }
                if (y > 0 && dil[i - ws] && lab[i - ws] == 0) { lab[i - ws] = nl; stack[sp++] = i - ws }
                if (y < hh - 1 && dil[i + ws] && lab[i + ws] == 0) { lab[i + ws] = nl; stack[sp++] = i + ws }
            }
            if (cnt < 8 || x1 < 0) continue
            val bw = x1 - x0 + 1; val bh = y1 - y0 + 1
            var why = ""
            if (bw * s < 16 || bh * s < 12) why = "작음"
            else if (bw > Wm2Params.maxW * ws) why = "너무넓음"
            else if (bh * s > Wm2Params.maxHpx) why = "너무높음"
            val dom = dominance(c, lab, nl, x0, y0, x1, y1, ws)
            if (why.isEmpty() && dom > Wm2Params.domMax) why = "직선꼴"
            out.add(Blob(x0, y0, x1, y1, cnt, sc, dom, why, lab, nl))
        }
        return out
    }

    private fun components(c: Cons, planes: List<Plane>, offs: IntArray, anchor: Double, s: Int, ws: Int, hmin: Int, tag: String, log: MutableList<String>?): List<Cand> {
        val k = planes.size; val hit = c.hit; val killed = IntArray(2); val rowKill = BooleanArray(hmin)
        val out = ArrayList<Cand>()
        // 전면 반복 무늬: (직선 제거 전) 반응이 이미지 전체 칸에 퍼져 있음
        run {
            val g = 6; var active = 0; var tot = 0; var sc = 0f
            for (gy in 0 until g) for (gx in 0 until g) {
                var n = 0; val x0 = gx * ws / g; val x1 = (gx + 1) * ws / g; val y0 = gy * hmin / g; val y1 = (gy + 1) * hmin / g
                for (y in y0 until y1) for (x in x0 until x1) if (hit[y * ws + x]) { n++; sc += c.sm[y * ws + x] }
                tot += n; if (n >= Wm2Params.tileCell * (x1 - x0) * (y1 - y0)) active++
            }
            if (active >= Wm2Params.tileFrac * g * g) out.add(Cand(anchor, 0, 0, ws, hmin, sc / 100f, tot, BooleanArray(k) { true }, offs, ws, hmin, "", 0f, sc / max(1, tot), "tile"))
        }
        // 가로 띠: 직선으로 지워질 두 평행선 사이에 내용(글자)이 있음
        val rk = BooleanArray(hmin); val hitCopy = hit.copyOf(); killLines(hitCopy, ws, hmin, rk)
        run {
            val rows = (0 until hmin).filter { rk[it] }
            var i = 0
            while (i < rows.size) {
                var j = i; while (j + 1 < rows.size && rows[j + 1] - rows[j] <= 2) j++
                val r1 = rows[j]
                var m = j + 1
                while (m < rows.size) {
                    var m2 = m; while (m2 + 1 < rows.size && rows[m2 + 1] - rows[m2] <= 2) m2++
                    val r2 = rows[m]; val gap = r2 - r1
                    if (gap * s in Wm2Params.bandMin..Wm2Params.bandMax) {
                        var inside = 0; var sc = 0f
                        for (y in r1 + 2 until r2 - 1) for (x in 0 until ws) if (hit[y * ws + x] && !rk[y]) { inside++; sc += c.sm[y * ws + x] }
                        if (inside >= Wm2Params.bandInside * ws * gap) out.add(Cand(anchor, 0, r1, ws, gap, sc / 100f, inside, BooleanArray(k) { true }, offs, ws, hmin, "", 0f, sc / max(1, inside), "band"))
                    }
                    m = m2 + 1
                }
                i = j + 1
            }
        }
        val bl = blobs(c, ws, hmin, s, killed)
        for (b in bl) {
            val present = BooleanArray(k)
            for (im in 0 until k) { var ok = 0; var tot = 0
                for (y in b.y0..b.y1) for (x in b.x0..b.x1) { val i = y * ws + x; if (hit[i] && b.lab[i] == b.id) { tot++; if (planes[im].mag[(y + offs[im]) * ws + x] >= 0.5f * Wm2Params.lowMag) ok++ } }
                present[im] = tot > 0 && ok >= 0.6 * tot }
            out.add(Cand(anchor, b.x0, b.y0, b.w, b.h, b.sc / 100f, b.cnt, present, offs, ws, hmin, b.why, b.dom, b.sc / b.cnt))
        }
        log?.add("정렬 %.2f%s: 직선 가로 %d 세로 %d줄 제거, 덩어리 %d, 통과 %d".format(anchor, tag, killed[0], killed[1], bl.size, out.count { it.why.isEmpty() }))
        return out
    }

    // ------------------------------------------------------------------ 진입점
    fun recognize(imgs: List<Img>): Wm2Result {
        val log = ArrayList<String>(); val marks = List(imgs.size) { ArrayList<Mark>() }
        val s = Wm2Params.scale
        val byW = HashMap<Int, ArrayList<Int>>()
        for (i in imgs.indices) byW.getOrPut(imgs[i].w) { ArrayList() }.add(i)
        for ((w, idx0) in byW) {
            val idx = dedupe(imgs, idx0)
            if (idx.size < 3) { log.add("폭 $w: 장 ${idx.size}개(중복 제거 후) — 판정 안 함"); continue }
            if (Wm2Params.mode >= 1) {
                val sp = Wm2Params.scalePair
                val pl = idx.map { prep(imgs[it], sp) }
                val cl = Wm2Pair.run(pl, sp, log).sortedByDescending { it.score }
                for (c in cl) { val f0 = c.items[0]
                    log.add("@cl\t%d\t%d\t%d\t%d\t%d\t%d\t%.2f\t%d\t%d\t%.2f\t%.3f\t%.3f\t%.3f\t%d".format(idx[c.ref], f0.x * sp, f0.y * sp, f0.w * sp, f0.h * sp, c.support, c.score, c.k, c.nProps, c.meanZ, c.dom, c.dens, c.spread, c.items.size)) }
                val need = max(2, Math.ceil(Wm2Params.pairFrac * (idx.size - 1)).toInt() + 1)
                val taken = ArrayList<Wm2Pair.Cluster>()
                for (c in cl) {
                    if (c.score < Wm2Params.minPair || c.support < need) continue
                    if (taken.any { t -> c.items.any { f -> t.items.any { g -> g.img == f.img && boxIou(f.x, f.y, f.w, f.h, g.x, g.y, g.w, g.h) >= 0.3f } } }) continue
                    if (taken.isNotEmpty() && c.score < Wm2Params.relBest * taken[0].score) continue
                    taken.add(c); if (taken.size >= 3) break
                }
                for (c in taken) for (f in c.items) {
                    val bx = f.x * sp; val by = f.y * sp; val bw = f.w * sp; val bh = f.h * sp
                    if (marks[idx[f.img]].any { m -> boxIou(m.x, m.y, m.w, m.h, bx, by, bw, bh) >= 0.4f }) continue
                    marks[idx[f.img]].add(Mark(bx, by, bw, bh, "pair", c.score))
                }
            }
            if (Wm2Params.mode != 1) {
            val planes = idx.map { prep(imgs[it], s) }
            val k = planes.size; val ws = planes[0].w; val hmin = planes.minOf { it.h }
            val all = ArrayList<Cand>()
            for (a in doubleArrayOf(0.0, 1.0, 0.25, 0.5, 0.75)) {
                val offs = IntArray(k) { (a * (planes[it].h - hmin)).toInt() }
                val nm = nullMap(planes, offs, ws, hmin)
                all.addAll(components(consensus(planes, offs, ws, hmin, 0, nm), planes, offs, a, s, ws, hmin, "", log))
            }
            val ok = all.filter { it.why.isEmpty() }.sortedByDescending { it.score }
            for (c in all) log.add("@cand\t%d\t%.2f\t%d\t%d\t%d\t%d\t%.3f\t%d\t%d\t%d\t%.3f\t%.3f\t%s".format(idx[0], c.anchor, c.x * s, c.y * s + c.offs[0] * s, c.w * s, c.h * s, c.score, c.hits, c.present.count { it }, idx.size, c.dom, c.meanR, c.why.ifEmpty { "-" } + (if (c.kind != "mark") "/" + c.kind else "")))
            val taken = ArrayList<Cand>()
            for (c in ok) {
                if (c.score < Wm2Params.minScore) continue
                if (c.present.count { it } < max(3.0, Math.ceil(Wm2Params.minPresent * idx.size))) continue
                if (taken.any { overlap(it, c, imgs[idx[0]].h) }) continue
                if (taken.isNotEmpty() && c.score < Wm2Params.relBest * taken[0].score) continue
                taken.add(c); if (taken.size >= 3) break
            }
            for (c in taken) for (j in idx.indices) if (c.present[j]) {
                val im = imgs[idx[j]]
                when (c.kind) {
                    "tile" -> marks[idx[j]].add(Mark(0, 0, im.w, im.h, "tile", c.score))
                    "band" -> marks[idx[j]].add(Mark(0, (c.y + c.offs[j]) * s, im.w, c.h * s, "band", c.score))
                    else -> marks[idx[j]].add(Mark(c.x * s, (c.y + c.offs[j]) * s, c.w * s, c.h * s, "mark", c.score))
                }
            }
            }
        }
        return Wm2Result(marks, log)
    }

    internal fun boxIou(ax: Int, ay: Int, aw: Int, ah: Int, bx: Int, by: Int, bw: Int, bh: Int): Float {
        val ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); val iy = max(0, min(ay + ah, by + bh) - max(ay, by)); val i = ix * iy; val u = aw * ah + bw * bh - i
        return if (u > 0) i.toFloat() / u else 0f
    }

    private fun overlap(a: Cand, b: Cand, h: Int): Boolean {
        val ay = a.y + a.offs[0]; val by = b.y + b.offs[0]
        val ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x)); val iy = max(0, min(ay + a.h, by + b.h) - max(ay, by))
        return ix * iy > 0.3 * min(a.w * a.h, b.w * b.h)
    }

    /** 같은 이미지(축소해 거의 같은 것) 제거 */
    private fun dedupe(imgs: List<Img>, idx: List<Int>): List<Int> {
        val keep = ArrayList<Int>(); val sig = ArrayList<IntArray>()
        for (i in idx) {
            val im = imgs[i]; val sg = IntArray(64)
            for (j in 0 until 64) { val x = (j % 8) * im.w / 8 + im.w / 16; val y = (j / 8) * im.h / 8 + im.h / 16; sg[j] = im.px[y * im.w + x] and 0xffffff }
            if (sig.none { it.contentEquals(sg) && imgs[keep[sig.indexOf(it)]].h == im.h }) { keep.add(i); sig.add(sg) }
        }
        return keep
    }
}
