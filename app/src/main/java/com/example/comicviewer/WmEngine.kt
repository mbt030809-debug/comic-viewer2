package com.example.comicviewer

import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.util.concurrent.Executors
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** 디코드된 전체 이미지(수정 가능) */
interface WmFull {
    val w: Int
    val h: Int
    fun getRegion(x: Int, y: Int, rw: Int, rh: Int, out: IntArray)
    fun setRegion(x: Int, y: Int, rw: Int, rh: Int, px: IntArray)
    fun encode(png: Boolean): ByteArray
    fun close()
}

/** 이미지 디코드/인코드(안드로이드는 BitmapFactory, PC 시험은 ImageIO) */
interface WmCodec {
    fun info(bytes: ByteArray): IntArray?            // [너비, 높이]
    fun small(bytes: ByteArray, s: Int): WmImg?      // 1/s 로 줄여 디코드
    fun full(bytes: ByteArray): WmFull?
}

/**
 * 워터마크 자동 감지·학습·제거 엔진.
 * 처리 흐름(이미지 한 장): 축소 윤곽 → 저장된/현재 모델과 대조 → (있으면) 학습 표본 추가 + 제거 / (없으면) 풀에 모아 반복 패턴 감지.
 * 어떤 단계든 실패하면 null(원본 그대로)을 돌려줌.
 */
class WmEngine(private val codec: WmCodec, private val file: File?) {
    private val lock = Any()
    @Volatile var enabled = true
    @Volatile private var giveUp = false
    private val models = ArrayList<WmModel>()      // 최근에 쓴 순서
    private var active: WmModel? = null
    private var misses = 0
    private class Item(val bytes: ByteArray, val thumb: WmThumb) { val key: Long = bytes.size * 31L + bytes.contentHashCode() }   // key: 같은 이미지를 다시 받은 경우 풀에서 중복 제거용
    private val pool = ArrayList<Item>()
    private var detecting = false
    private var attempts = 0
    private var sinceTry = 0
    private var seen = 0
    private var cleanedPage = 0
    private var dirty = false
    @Volatile var lastGuard = ""   // (시험용) 마지막 지우기 전/후 윤곽 일치도
    @Volatile var logoL = 140      // 로고 색 밝기(0~255): 그림 위 보정에 사용
    @Volatile var gain = 100       // 그림 위 보정 세기(%): 단색 배경에서 본 워터마크를 반대로 입히는 정도
    @Volatile var lastErr = ""     // 마지막 오류(진단용)
    private val samples = ArrayList<ByteArray>()   // 진단용: 최근에 검사한 원본 이미지(워터마크가 찍힌 채 받은 것) 최대 8장
    fun sampleImages(): List<ByteArray> = synchronized(samples) { ArrayList(samples) }
    @Volatile private var eligible = 0     // 크기 조건을 통과해 검사한 이미지 수
    @Volatile private var cleanedTotal = 0 // 지운 이미지 수(전체)
    @Volatile private var cRing = 0; @Volatile private var cInterior = 0; @Volatile private var cNoChange = 0; @Volatile private var cNn = 0; @Volatile private var cPres = 0
    @Volatile private var lastInfo = ""
    private val logLines = ArrayList<String>()   // 진단용: 이미지·찾기마다 무엇을 어떻게 판단했는지(최근 60줄)
    private var imgSeq = 0
    @Volatile var logSink: ((String) -> Unit)? = null   // 판단 기록을 바깥(logcat)으로도 내보냄
    private fun note(t: String) { synchronized(logLines) { logLines.add(t); if (logLines.size > 60) logLines.removeAt(0) }; try { logSink?.invoke(t) } catch (_: Throwable) { } }
    @Volatile var nn: WmNet? = null              // 신경망(없으면 계산식만 사용)
    @Volatile var onModel: (() -> Unit)? = null   // 새 워터마크 모양을 처음 찾았을 때(이미 지나간 이미지를 다시 받아 지우는 데 씀)
    private val BAD_MAX = 5   // 모양 불일치로 연속 이만큼 취소되면 그 모델은 잠시 제외
    @Volatile private var skippedGuard = 0 // 지웠다가 효과 없어 취소한 수
    private val exec = Executors.newSingleThreadExecutor { r -> Thread(r).apply { priority = Thread.MIN_PRIORITY; isDaemon = true } }

    // 이전에 저장한 모양은 불러오지 않음: 예전 화·예전 워터마크의 모델이 남아 새로 찾는 것을 막고(현재 모델이 있으면 찾기를 안 함) 엉뚱한 자리에 점을 찍던 문제(기기 시험에서 확인)

    /** (시험용) 가장 최근 모델과의 윤곽 일치도 */
    fun debugPresence(bytes: ByteArray): Float {
        val dim = codec.info(bytes) ?: return -9f
        val s = WmCore.scaleFor(dim[0]); val small = codec.small(bytes, s) ?: return -9f
        val th = WmCore.makeThumb(small, s, dim[0], dim[1])
        val m = synchronized(lock) { models.firstOrNull { it.w == dim[0] } } ?: return -8f
        return WmCore.presence(m, th)
    }

    /** 진단용 한 줄: 검사한 이미지 수, 지운 수, 마지막 오류 */
    fun diag(): String = "검사 $eligible · 지움 $cleanedTotal" + (if (cNn > 0) " · 신경망 $cNn" else "") + (if (skippedGuard > 0) " · 취소 $skippedGuard" else "") + (if (lastErr.isNotEmpty()) "\n오류: $lastErr" else "")

    /** 진단 전체(클립보드 복사용) */
    fun fullDiag(): String = synchronized(lock) {
        val sb = StringBuilder()
        sb.append("■ 상태: ").append(status()).append("\n").append(diag()).append("\n")
        sb.append("■ 이번 페이지: 검사 ${eligible}장 · 모델 ${models.size}개 · 찾기 시도 ${attempts}회(최대 24) · 풀 ${pool.size}장 · 있음 판정 $cPres · 둘레 단색 아님 $cRing · 상자 테두리 단색 아님 $cInterior · 변화 없음 $cNoChange")
        sb.append(if (giveUp) " · 찾기 중단됨" else "").append("\n")
        sb.append("■ 마지막 지우기: ").append(lastInfo).append("\n")
        if (models.isEmpty()) sb.append("■ 모델: 없음(아직 워터마크 모양을 못 찾음)\n")
        for ((k, m) in models.withIndex()) sb.append("■ 모델$k: ").append(if (m.frame == WmCore.FREE) "세로위치 자유" else "프레임${m.frame}").append(" · 이미지너비 ${m.w} · 상자 ${m.nbw}x${m.nbh}px @x${m.nx} · 지움 ${m.cleaned}장 · 표본 ${m.n}/그림 ${m.aN} · 다양성 %.0f".format(m.diversity)).append(if (m.learned) (if (m.approx) " · 학습(근사)" else " · 학습") else " · 미학습").append("\n")
        sb.append("■ 최근 판단 기록(오래된 것 → 최근, 최대 60줄):\n")
        synchronized(logLines) { for (l in logLines) sb.append(l).append("\n") }
        sb.toString()
    }

    /** 이 페이지에서 이미지를 가로챌 필요가 있는가(꺼졌거나, 여러 번 찾아도 없으면 false → 웹뷰가 직접 받음) */
    @Volatile var marker = true    // 인식 표시(가운데 빨간 점): 지우기와 별개로 켜고 끔
    fun wants() = (enabled || marker) && !giveUp

    /** 페이지 이동: 이번 화의 임시 상태 초기화(저장된 모델은 유지) */
    /** 새 페이지를 열 때마다 '다시 찾기'와 같이 처음부터: 저장된 모양을 버리고 이번 페이지의 이미지들에서 새로 찾음(바뀐 워터마크에 낡은 모양이 남지 않게) */
    fun onPage() {
        synchronized(samples) { samples.clear() }
        synchronized(lock) { models.clear(); pending = null; pool.clear(); attempts = 0; sinceTry = 0; giveUp = false; misses = 0; active = null; seen = 0; cleanedPage = 0; dirty = false }
    }

    /** 저장된 모델과 이번 화의 상태를 모두 버리고 처음부터 다시 찾음 */
    fun forget() {
        synchronized(lock) { models.clear(); active = null; pool.clear(); attempts = 0; sinceTry = 0; giveUp = false; misses = 0; dirty = false }
        try { file?.delete() } catch (_: Exception) {}
    }

    fun status(): String = synchronized(lock) {
        val a = active
        when {
            !enabled -> "꺼짐"
            a != null -> if (a.learned) "지우는 중 · ${cleanedPage}장" else "감지됨 · 배경 학습 ${min(100, (a.diversity * 100 / WmCore.LEARN_D).toInt())}%"
            giveUp -> "워터마크 없음 · 반복 상관 %.2f".format(WmCore.lastMax)
            detecting || pool.isNotEmpty() -> "찾는 중 · ${pool.size}장 · 반복 상관 %.2f".format(WmCore.lastMax)
            models.isNotEmpty() -> "대기 · 저장된 모양 ${models.size}개"
            else -> "대기"
        }
    }

    /** 이미지 한 장 처리. 지운 결과(바이트)를 돌려주고, 바꿀 게 없거나 실패하면 null */
    fun process(bytes: ByteArray, mime: String): ByteArray? {
        if (!wants()) return null
        try {
            val dim = codec.info(bytes) ?: run { lastErr = "이미지 해석 실패"; return null }
            val w = dim[0]; val h = dim[1]
            if (w < 350 || h < 100 || w > 4000 || h > 30000 || w.toLong() * h > 16_000_000L || w.toFloat() / h > 8f) return null
            eligible++
            synchronized(samples) { samples.add(bytes); if (samples.size > 8) samples.removeAt(0) }
            val s = WmCore.scaleFor(w)
            val small = codec.small(bytes, s) ?: run { lastErr = "축소 디코드 실패"; return null }
            val th = WmCore.makeThumb(small, s, w, h)
            var model: WmModel? = null; var loc = floatArrayOf(0f, 0f, 0f)
            var noteTxt = ""
            synchronized(lock) {
                seen++
                val cands = ArrayList<WmModel>()
                active?.let { if (it.w == w && it.badStreak < BAD_MAX) cands.add(it) }
                for (m in models) if (m.w == w && m !== active && m.badStreak < BAD_MAX) cands.add(m)
                var best = -1f; val per = StringBuilder()
                for ((ci, m) in cands.withIndex()) { val r = WmCore.locate(m, th); val p = if (m.frame == WmCore.FREE) r[0] - 0.04f else r[0]; if (p > best) { best = p; model = m; loc = r }
                    per.append(" 모델${models.indexOf(m)}(").append(if (m.frame == WmCore.FREE) "자유" else "프레임${m.frame}").append(" 상자${m.nbw}x${m.nbh}@x${m.nx}) 일치%.2f".format(r[0])).append(if (m.frame == WmCore.FREE) " 위치y${r[2].toInt() * m.s}" else "") }
                val hit = model
                        // 위치가 정해진 모양(위/아래/비율 기준)은 이미 두 장 이상 지워 확인됐으면 일치 기준을 낮춤: 사람처럼 '그 자리에 늘 있는 것'으로 보고 판정
                val thr = if (hit != null && hit.frame != WmCore.FREE && hit.cleaned >= 2) 0.07f else WmCore.PRES_MIN
                noteTxt = "#${++imgSeq} ${w}x${h} ${th.info}" + (if (cands.isEmpty()) " · 비교할 모델 없음" else " ·" + per + " (기준 %.2f)".format(thr))
                if (hit != null && best >= thr) {
                    active = hit; misses = 0; models.remove(hit); models.add(0, hit)
                    noteTxt += " → 인식"
                } else {
                    model = null
                    noteTxt += " → 미인식, 찾기용 풀에 보관(풀 ${pool.size + 1}장)"
                    if (active != null && cands.isNotEmpty() && ++misses >= 3) { active = null; misses = 0 } // 연속으로 안 맞으면 바뀐 것
                    pool.removeAll { it.thumb.w != w }
                    run { val it0 = Item(bytes, th); pool.removeAll { q -> q.key == it0.key }; pool.add(it0) }; if (pool.size > 8) pool.removeAt(0)
                    sinceTry++
                    if (active == null && !detecting && attempts < 24 && pool.size >= 3 && sinceTry >= (if (attempts == 0) 3 else 1)) {
                        detecting = true; attempts++; sinceTry = 0
                        val items = ArrayList(pool)
                        exec.execute { runDetect(items) }
                    }
                }
            }
            note(noteTxt)
            val m = model ?: return null
            val bad = BooleanArray(1)
            val out = cleanWith(m, bytes, mime, w, h, loc, bad)
            if (bad[0]) { // 모양이 안 맞아 취소됨: 반복되면 워터마크가 바뀐 것 → 이 모델을 잠시 제외하고 이 이미지를 새로 찾는 데 씀
                synchronized(lock) {
                    m.badStreak++
                    if (m.badStreak >= BAD_MAX && active === m) { active = null; misses = 0 }
                    pool.removeAll { it.thumb.w != w }
                    run { val it0 = Item(bytes, th); pool.removeAll { q -> q.key == it0.key }; pool.add(it0) }; if (pool.size > 8) pool.removeAt(0)
                    sinceTry++
                    if (active == null && !detecting && attempts < 24 && pool.size >= 3 && sinceTry >= (if (attempts == 0) 3 else 1)) {
                        detecting = true; attempts++; sinceTry = 0
                        val items = ArrayList(pool)
                        exec.execute { runDetect(items) }
                    }
                }
            } else if (out != null) m.badStreak = 0
            return out
        } catch (t: Throwable) { lastErr = (t.javaClass.simpleName + ":" + (t.message ?: "")).take(80); return null }
    }

    private var pending: WmDet? = null   // 약한 증거로 나온 후보(다음 시도에서 같은 후보가 또 나오면 확정)

    private fun sameCand(a: WmDet, b: WmDet): Boolean =
        a.frame == b.frame && abs(a.tx0 - b.tx0) <= 3 && abs(a.tbw - b.tbw) <= max(3, a.tbw / 4) && abs(a.tbh - b.tbh) <= max(3, a.tbh / 4) &&
            (a.frame == WmCore.FREE || abs(a.tyF - b.tyF) <= 4)

    @Volatile private var buildWhy = ""

    private fun runDetect(items: List<Item>) {
        var built: WmModel? = null
        val sb = StringBuilder("찾기 #$attempts: 풀 ${items.size}장(단색 ${items.count { it.thumb.calm }}장)")
        try {
            val ths = items.map { it.thumb }
            var det = WmCore.detect(ths); var df: WmDet? = null
            sb.append(" | ").append(WmCore.dbgDet)
            if (det == null) { df = WmCore.detectFree(ths); sb.append(" | 가변위치형: ").append(if (df != null) "후보 있음" else WmCore.dbgFree) }
            var weak = false
            if (det == null && df == null) { // 약한 증거: 연속 두 번 같은 자리에서 나오면 확정(이미지가 쌓이며 표가 모이는 것과 같음)
                det = WmCore.detect(ths, 0.7f)
                if (det == null) df = WmCore.detectFree(ths, 0.7f)
                weak = det != null || df != null
                sb.append(" | 약한 증거 재시도: ").append(if (weak) "후보 있음" else "없음")
            }
            val cand = det ?: df
            var accept = cand != null
            if (weak && cand != null) {
                synchronized(lock) { val p = pending; if (p != null && sameCand(p, cand)) pending = null else { pending = cand; accept = false } }
                if (!accept) sb.append(" → 약한 후보라 다음 시도에서 같은 자리가 또 나오면 확정")
            }
            if (cand != null) sb.append(" | 후보 ").append(if (cand.frame == WmCore.FREE) "자유" else "프레임${cand.frame}").append(" 상자 ${cand.tbw * ths[0].s}x${cand.tbh * ths[0].s}@x${cand.tx0 * ths[0].s}")
            if (accept) {
                buildWhy = ""
                built = if (det != null) build(det, items) else if (df != null) buildFree(df, items) else null
                sb.append(if (built != null) " → 모델 생성됨" else " → 모델 만들기 실패: " + buildWhy.ifEmpty { "표본 부족" })
            } else if (cand == null) sb.append(" → 후보 없음")
        } catch (t: Throwable) { sb.append(" → 오류 ").append(t.javaClass.simpleName).append(":").append(t.message ?: ""); lastErr = ("찾기 오류 " + t.javaClass.simpleName + ":" + (t.message ?: "")).take(80) }
        note(sb.toString())
        synchronized(lock) {
            detecting = false
            if (built != null) {
                models.add(0, built); while (models.size > 4) models.removeAt(models.size - 1)
                active = built; misses = 0; pool.clear(); attempts = 0; sinceTry = 0; dirty = true; pending = null
            } else if (attempts >= 24) { giveUp = true; note("찾기 24회 실패 → 이 페이지는 워터마크 없음으로 판단하고 중단") }
        }
        if (built != null) { persist(); try { onModel?.invoke() } catch (_: Throwable) { } }
    }

    private fun build(det: WmDet, items0: List<Item>): WmModel? {
        val s = items0[0].thumb.s; val w = items0[0].thumb.w; val tw = items0[0].thumb.tw
        val pad = WmCore.REG_PAD
        val minH = items0.minOf { it.thumb.h }
        val nbw = min(det.tbw * s, w - det.tx0 * s)
        val nbh = min(det.tbh * s, minH - det.tyF * s)
        if (nbw < 40 || nbh < 40) { buildWhy = "상자가 너무 작음(${nbw}x${nbh}px)"; return null }
        fun y0n(h: Int) = WmCore.nativeYOf(det.frame, det.tyF * s, nbh, h, det.tref, s)
        val items = items0.filter { y0n(it.thumb.h) >= 0 && y0n(it.thumb.h) + nbh <= it.thumb.h }
        if (items.size < 3) { buildWhy = "상자가 이미지 안에 있는 풀 이미지가 ${items.size}장뿐"; return null }
        // 이미지 가장자리에서는 여백이 줄어들 수 있음(모든 조각의 크기를 같게 맞춤)
        val padL = min(pad, det.tx0 * s); val padR = min(pad, w - (det.tx0 * s + nbw))
        val padT = min(pad, items.minOf { y0n(it.thumb.h) }); val padB = min(pad, items.minOf { it.thumb.h - (y0n(it.thumb.h) + nbh) })
        val rw = nbw + padL + padR; val rh = nbh + padT + padB
        val regs = ArrayList<WmReg>()
        for (it in items) {
            val full = codec.full(it.bytes) ?: continue
            try {
                val h = full.h
                val x0 = det.tx0 * s - padL; val y0 = y0n(h) - padT
                if (x0 < 0 || y0 < 0 || x0 + rw > full.w || y0 + rh > h) continue
                val px = IntArray(rw * rh); full.getRegion(x0, y0, rw, rh, px)
                regs.add(WmReg(px, x0, y0, h))
            } finally { full.close() }
        }
        return WmCore.buildModel(det, s, w, tw, nbw, nbh, regs, rw, rh, padL, padT)
    }

    /** 세로 위치가 제각각인 워터마크: 풀의 각 이미지에서 찾은 위치(det.locs)로 상자 조각을 떼어 서로 맞춘 뒤 모델 생성 */
    private fun buildFree(det: WmDet, items: List<Item>): WmModel? {
        val locs = det.locs ?: return null
        val s = items[0].thumb.s; val w = items[0].thumb.w; val tw = items[0].thumb.tw
        val pad = WmCore.REG_PAD; val m = s + 2
        val x0n = det.tx0 * s; val nbw = min(det.tbw * s, w - x0n); val nbh = det.tbh * s
        if (nbw < 40 || nbh < 40) { buildWhy = "상자가 너무 작음(${nbw}x${nbh}px)"; return null }
        val left = x0n; val right = w - (x0n + nbw)
        val mx = minOf(m, left, right).coerceAtLeast(0)
        val padL = min(pad, left - mx).coerceAtLeast(0); val padR = min(pad, right - mx).coerceAtLeast(0)
        val rw = nbw + padL + padR; val rh = nbh + 2 * pad
        val regs = ArrayList<WmReg>(); var ref: FloatArray? = null
        for (i in items.indices) {
            if (locs[i] < 0) continue
            val it = items[i]; val y0e = locs[i] * s
            val full = codec.full(it.bytes) ?: continue
            try {
                val h = full.h
                if (y0e - pad - m < 0 || y0e + nbh + pad + m > h) continue
                val bw = rw + 2 * mx; val bh = rh + 2 * m
                val bx = x0n - padL - mx; val by = y0e - pad - m
                val big = IntArray(bw * bh); full.getRegion(bx, by, bw, bh, big)
                val g = WmCore.colorGradMag(big, bw, bh)
                var ox = mx; var oy = m
                val r0 = ref
                if (r0 == null) {
                    ref = FloatArray(rw * rh) { g[(oy + it / rw) * bw + ox + it % rw] }
                } else {
                    var sb = 0.0; var sbb = 0.0; for (v in r0) { sb += v; sbb += v.toDouble() * v }
                    val cnt = r0.size; val vb = sbb - sb * sb / cnt
                    var best = -2.0
                    for (dy in -m..m) for (dx in -mx..mx) {
                        var sa = 0.0; var saa = 0.0; var sab = 0.0
                        var k = 0
                        while (k < cnt) { val a = g[(m + dy + k / rw) * bw + mx + dx + k % rw].toDouble(); sa += a; saa += a * a; sab += a * r0[k]; k += 2 }
                        val c2 = cnt / 2.0; val va = saa - sa * sa / c2
                        if (va < 1e-6) continue
                        val sb2 = sb / 2; val vb2 = vb / 2
                        val c = (sab - sa * sb2 / c2) / sqrt(va * vb2)
                        if (c > best) { best = c; ox = mx + dx; oy = m + dy }
                    }
                }
                val px = IntArray(rw * rh)
                for (y in 0 until rh) System.arraycopy(big, (oy + y) * bw + ox, px, y * rw, rw)
                regs.add(WmReg(px, bx + ox, by + oy, h))
            } finally { full.close() }
        }
        if (regs.size < 2) { buildWhy = "상자 조각을 뗀 이미지가 ${regs.size}장뿐"; return null }
        return WmCore.buildModel(det, s, w, tw, nbw, nbh, regs, rw, rh, padL, pad)
    }

    /** 인식한 워터마크 상자의 가운데에 빨간 점(지름 약 14px)을 찍음: 위치를 제대로 인식했는지 눈으로 확인용 */
    private fun drawDot(full: WmFull, w: Int, h: Int, cx: Int, cy: Int) {
        val r = 7; val x0 = max(0, cx - r); val y0 = max(0, cy - r); val x1 = min(w, cx + r + 1); val y1 = min(h, cy + r + 1)
        if (x1 <= x0 || y1 <= y0) return
        val dw = x1 - x0; val dh = y1 - y0; val px = IntArray(dw * dh)
        for (y in 0 until dh) for (x in 0 until dw) { val dx = x0 + x - cx; val dy = y0 + y - cy; px[y * dw + x] = if (dx * dx + dy * dy <= r * r) 0xFFE00000.toInt() else 0 }
        val cur = IntArray(dw * dh); full.getRegion(x0, y0, dw, dh, cur)
        for (i in px.indices) if (px[i] != 0) cur[i] = px[i]
        full.setRegion(x0, y0, dw, dh, cur)
    }

    private fun cleanWith(m: WmModel, bytes: ByteArray, mime: String, w: Int, h: Int, loc: FloatArray, bad: BooleanArray): ByteArray? {
        if (mime.contains("png") && w.toLong() * h > 4_000_000L) return null // 큰 PNG 는 다시 저장하는 데 너무 오래 걸림
        val full = codec.full(bytes) ?: return null
        try {
            val free = m.frame == WmCore.FREE
            val rad = if (free) m.s + 1 else 2
            var x0 = if (free) m.nx + loc[1].toInt() * m.s else m.nx
            var y0 = if (free) loc[2].toInt() * m.s else m.nativeY(h)
            val mg = 12 + rad // 둘레 검사(12px) + 위치 맞춤 여백
            // 상자가 이미지 가장자리에 걸쳐도(워터마크가 아래/위쪽 끝에 붙은 경우) 처리: 밖은 가장자리 값을 복제해 채우고, 지운 뒤 안쪽만 되돌려 씀
            fun overlap(): Int { val iw = min(w, x0 + m.nbw) - max(0, x0); val ih = min(h, y0 + m.nbh) - max(0, y0); return if (iw <= 0 || ih <= 0) 0 else iw * ih }
            if (overlap() * 10 < m.nbw * m.nbh * 6) return null   // 상자의 60% 이상이 이미지 안에 있어야 함
            val pw = m.nbw + 2 * mg; val ph = m.nbh + 2 * mg
            fun readPatch(): IntArray {
                val gx0 = x0 - mg; val gy0 = y0 - mg
                val rx0 = max(0, gx0); val ry0 = max(0, gy0); val rx1 = min(w, gx0 + pw); val ry1 = min(h, gy0 + ph)
                if (rx1 <= rx0 || ry1 <= ry0) return IntArray(pw * ph)
                val rw = rx1 - rx0; val rh = ry1 - ry0
                val raw = IntArray(rw * rh); full.getRegion(rx0, ry0, rw, rh, raw)
                val out = IntArray(pw * ph)
                for (y in 0 until ph) { val sy = (gy0 + y).coerceIn(ry0, ry1 - 1) - ry0
                    for (x in 0 until pw) out[y * pw + x] = raw[sy * rw + ((gx0 + x).coerceIn(rx0, rx1 - 1) - rx0)] }
                return out
            }
            var patch = readPatch()
            if (m.frame != WmCore.TL && m.frame != WmCore.BL) { // 비율/자유 위치는 반올림·축소 때문에 어긋날 수 있음
                val al = m.align(patch, pw, ph, mg, mg, rad)
                if (al[0] != 0 || al[1] != 0) { x0 += al[0]; y0 += al[1]; if (overlap() * 10 < m.nbw * m.nbh * 6) return null; patch = readPatch() }
            }
            if (!enabled) { // 지우기가 꺼져 있으면 지우지 않고 인식 표시만
                if (!marker) return null
                cPres++; drawDot(full, w, h, x0 + m.nbw / 2, y0 + m.nbh / 2); return full.encode(mime.contains("png"))
            }
            val ox = mg; val oy = mg
            val region = IntArray(m.size)
            for (y in 0 until m.nbh) System.arraycopy(patch, (oy + y) * pw + ox, region, y * m.nbw, m.nbw)
            cPres++
            lastInfo = "프레임${m.frame} 일치 %.2f 위치(%d,%d) 이미지 %dx%d".format(loc[0], x0, y0, w, h)
            var flat = WmCore.ringFlat(patch, pw, ph, ox, oy, m.nbw, m.nbh)
            if (flat == null) cRing++ else if (!m.interiorFlat(region, flat)) { cInterior++; flat = null }
            if (flat != null) synchronized(lock) { m.addFlat(region, flat); m.solve(); dirty = true }
            else synchronized(lock) { m.addArt(region); if (m.aN % 5 == 0) { m.solve(); dirty = true } }
            val before = WmCore.regionPresence(m, region)
            val changed = m.clean(region, flat, gain / 100f, logoL.toFloat(), nn)
            val dotX = x0 + m.nbw / 2; val dotY = y0 + m.nbh / 2
            if (!changed) { cNoChange++; if (!marker) return null; drawDot(full, w, h, dotX, dotY); return full.encode(mime.contains("png")) }
            if (m.usedNn) cNn++
            val after = WmCore.regionPresence(m, region)
            lastGuard = "before=%.2f after=%.2f".format(before, after)
            if (if (m.usedNn) after > 0.9f * before + 0.02f else after > 0.7f * before + 0.03f) { skippedGuard++; bad[0] = true; if (!marker) return null; drawDot(full, w, h, dotX, dotY); return full.encode(mime.contains("png")) } // 지웠는데도 윤곽이 줄지 않으면 잘못 맞춘 것: 원본 유지
            run { // 이미지 안쪽 부분만 되돌려 씀
                val ix0 = max(0, x0); val iy0 = max(0, y0); val ix1 = min(w, x0 + m.nbw); val iy1 = min(h, y0 + m.nbh)
                val iw = ix1 - ix0; val ih = iy1 - iy0
                if (iw == m.nbw && ih == m.nbh) full.setRegion(x0, y0, m.nbw, m.nbh, region)
                else { val sub = IntArray(iw * ih); for (y in 0 until ih) System.arraycopy(region, (iy0 - y0 + y) * m.nbw + (ix0 - x0), sub, y * iw, iw); full.setRegion(ix0, iy0, iw, ih, sub) }
            }
            synchronized(lock) { m.cleaned++; cleanedPage++; cleanedTotal++ }
            if (marker) drawDot(full, w, h, dotX, dotY)
            return full.encode(mime.contains("png"))
        } finally { full.close() }
    }

    // ---------------------------------------------------------------- 저장/불러오기
    fun persist() {
        return   // 모양을 저장하지 않음(위 설명 참고)
        @Suppress("UNREACHABLE_CODE")
        val f = file ?: return
        val snap: List<WmModel> = synchronized(lock) { if (!dirty) return; dirty = false; models.map { it.copy() } }
        exec.execute {
            try {
                val bo = ByteArrayOutputStream()
                DataOutputStream(GZIPOutputStream(bo)).use { o -> o.writeInt(snap.size); for (m in snap) m.write(o) }
                val tmp = File(f.parentFile, f.name + ".tmp"); tmp.writeBytes(bo.toByteArray()); tmp.renameTo(f)
            } catch (_: Throwable) { }
        }
    }

    private fun load() {
        val f = file ?: return
        try {
            if (!f.exists()) return
            DataInputStream(GZIPInputStream(BufferedInputStream(FileInputStream(f)))).use { i ->
                val n = i.readInt()
                for (k in 0 until min(n, 4)) { val m = WmModel.read(i) ?: break; synchronized(lock) { if (models.size < 4) models.add(m) } }
            }
        } catch (_: Throwable) { }
    }
}
