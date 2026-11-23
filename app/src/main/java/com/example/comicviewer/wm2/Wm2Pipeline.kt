package com.example.comicviewer.wm2

import com.example.comicviewer.WmCodec
import com.example.comicviewer.WmFull
import kotlin.math.max
import kotlin.math.min

/**
 * 워터마크 처리의 전체 흐름. 이미지 한 장마다 한 방향으로만 흐른다:
 *   디코드 → [예시] 갤러리(흰/검정 배경의 원본 워터마크 이미지)면 예시 마크로 저장하고 끝
 *   → [인식] 예시 마크마다 Wm2Detector → [확정] Wm2Confirmer → [제거] Wm2Eraser → (표시) 빨간 상자 → 인코드
 * 예시(갤러리)가 아직 안 나왔으면 인식하지 않고 원본 그대로 둔다. 예시가 처음 나오는 순간 앞서 지나간 이미지를 다시 받게 한다.
 * 지운 뒤 다시 인식하는 단계는 없다. 각 단계는 서로의 내부를 모른다.
 */
class Wm2Pipeline(private val codec: WmCodec) {
    class Analysis(val confirmed: List<Wm2Detector.Box>, val p: FloatArray, val w2: Int, val h2: Int)
    /** 예시 마크 하나: 모델 입력(ref)과 원래 크기(w,h). 인식된 상자는 이 크기의 0.6~1.75배 안이어야 함 */
    class Example(val ref: FloatArray, val w: Int, val h: Int)
    private val exLock = Any()
    @Volatile private var examples: List<Example> = emptyList()   // 이 페이지의 예시 마크
    @Volatile private var fromGallery = false                    // 갤러리 이미지에서 얻은 예시가 있으면 여백 탐색은 하지 않음
    private var galleryKey = 0L
    private val s1Done = HashMap<Long, List<IntArray>>()         // 이미지 → 1단계 후보(여백에서 찾은 예시 후보 상자). 같은 이미지를 다시 받아도 다시 계산하지 않음
    val refCount: Int get() = examples.size
    @Volatile var stage1: Wm2Stage1? = null                      // 1단계 신경망(없으면 여백 예시 탐색 안 함)

    @Volatile var detector: Wm2Detector? = null      // 모델이 준비되기 전에는 아무것도 하지 않음
    @Volatile var erase = true                       // 확정된 워터마크 지우기
    @Volatile var showBoxes = false                  // 확정된 자리에 빨간 상자 표시
    @Volatile var logSink: ((String) -> Unit)? = null
    @Volatile var onConfirmed: (() -> Unit)? = null  // 예시가 처음 나왔을 때·처음 확정될 때: 앞서 원본으로 나간 이미지를 다시 받게 함
    @Volatile var lastErr = ""
    @Volatile private var seen = 0
    @Volatile private var changed = 0
    @Volatile private var lastMs = 0L
    private val confirmer = Wm2Confirmer()
    private val log = ArrayList<String>()

    fun wants(): Boolean = detector != null && (erase || showBoxes)
    /** 지금 설정으로 만든 결과를 구분하는 값(지우기·표시 조합). 캐시 키에 넣어 다른 설정으로 만든 이미지가 섞이지 않게 함 */
    fun flagsKey(): Int = (if (erase) 1 else 0) or (if (showBoxes) 2 else 0)
    fun onPage() { confirmer.reset(); synchronized(exLock) { examples = emptyList(); fromGallery = false; galleryKey = 0L; s1Done.clear() } }
    fun forget() = onPage()

    private fun note(t: String) { synchronized(log) { log.add(t); if (log.size > 60) log.removeAt(0) }; try { logSink?.invoke(t) } catch (_: Throwable) { } }
    fun status(): String {
        val flags = "[엔진 상태] 지우기 ${if (erase) "켜짐" else "꺼짐"} · 위치 표시 ${if (showBoxes) "켜짐" else "꺼짐"}\n"
        return flags + when {
            detector == null -> "모델 준비 중"
            examples.isEmpty() -> "예시 마크 기다리는 중(흰/검정 여백에 원본 워터마크가 있는 이미지가 나오면 시작) · 검사 ${seen}장"
            !erase && !showBoxes -> "꺼짐"
            confirmer.everConfirmed -> "예시 ${examples.size}개 · 워터마크 확정 · 처리 ${changed}장 · 검사 ${seen}장 · ${lastMs}ms/장"
            else -> "예시 ${examples.size}개 · 찾는 중 · 검사 ${seen}장(같은 너비 ${confirmer.size}장, 3장 이상이면 판정) · ${lastMs}ms/장"
        }
    }
    fun diag(): String = synchronized(log) { log.takeLast(12).joinToString("\n") }
    fun fullDiag(): String = status() + (if (lastErr.isNotEmpty()) "\n마지막 오류: $lastErr" else "") + "\n" + synchronized(log) { log.joinToString("\n") }

    /** 인식 + 확정만 한다(픽셀은 건드리지 않음). 이미지 조건에 안 맞으면 null. JVM 시험도 이 함수를 쓴다 */
    fun analyze(bytes: ByteArray): Analysis? {
        val det = detector ?: return null
        val dim = codec.info(bytes) ?: run { lastErr = "이미지 해석 실패"; return null }
        val w = dim[0]; val h = dim[1]
        if (w < 350 || h < 100 || w > 4000 || h > 30000 || w.toLong() * h > 16_000_000L || w.toFloat() / h > 8f) return null
        val t0 = System.currentTimeMillis()
        val img = codec.small(bytes, 1) ?: run { lastErr = "디코드 실패"; return null }
        val key = bytes.size * 31L + bytes.contentHashCode()
        // 갤러리(예시 마크 이미지)면 예시로 저장하고 이 장은 건드리지 않음
        val g = Wm2Gallery.find(img)
        if (g != null) {
            var fire = false
            synchronized(exLock) {
                if (key != galleryKey) {
                    examples = g.map { Example(Wm2Gallery.makeRef(img, it), it.w, it.h) }; galleryKey = key; fromGallery = true; fire = true
                    note("예시 마크 ${g.size}개 확보: 갤러리 ${w}x$h")
                }
            }
            if (fire) try { onConfirmed?.invoke() } catch (_: Throwable) { }
            return null
        }
        // 1단계: 갤러리가 없으면 이미지 속 단색 여백에서 예시 후보를 찾는다(찾은 후보는 예시로 등록, 그 자리는 인식에서 제외)
        val cands = if (fromGallery) emptyList() else stage1Candidates(key, img, w, h)
        val ex = examples
        if (ex.isEmpty()) { seen++; return null }
        val d = det.detect(img, ex.map { it.ref }, ex.map { intArrayOf(it.w, it.h) })
        val boxes = d.boxes.filter { b -> cands.none { c -> Wm2Detector.iou(b, Wm2Detector.Box(c[0], c[1], c[2], c[3], false, 0f)) > 0.3f } }
        lastMs = System.currentTimeMillis() - t0; seen++
        val r = confirmer.confirm(key, w, h, boxes)
        note("#$seen ${w}x$h 후보 ${boxes.size}개 · 비교 ${r.others}장" + (if (r.others < Wm2Confirmer.MIN_OTHERS) " → 판정 대기" else " → 확정 ${r.confirmed.size}개"))
        if (r.firstTime) try { onConfirmed?.invoke() } catch (_: Throwable) { }
        return Analysis(r.confirmed, d.p, d.w2, d.h2)
    }

    /** 이미지의 단색 여백에서 예시 후보를 찾아 돌려주고(규칙 → 신경망으로 거름), 처음 보는 이미지면 후보를 예시로 등록한다 */
    private fun stage1Candidates(key: Long, img: com.example.comicviewer.WmImg, w: Int, h: Int): List<IntArray> {
        synchronized(exLock) { s1Done[key]?.let { return it } }
        val s1 = stage1 ?: return emptyList()
        val rule = Wm2Examples.find(img)
        val c = if (rule.isEmpty()) emptyList() else s1.filter(s1.mask(img), rule)
        var fire = false
        synchronized(exLock) {
            if (s1Done.containsKey(key)) return s1Done[key]!!
            s1Done[key] = c
            if (c.isNotEmpty() && !fromGallery) {
                val add = c.take(MAX_EXAMPLES - examples.size).map { Example(Wm2Examples.makeRef(img, it), it[2], it[3]) }
                if (add.isNotEmpty()) { examples = examples + add; fire = true; note("예시 후보 ${add.size}개 등록: 여백 ${w}x$h → 예시 ${examples.size}개") }
            }
        }
        if (fire) try { onConfirmed?.invoke() } catch (_: Throwable) { }
        return c
    }

    /** 이미지 한 장 처리. 바뀐 바이트를 돌려주고, 확정 전이거나 바꿀 게 없으면 null(원본 그대로) */
    fun process(bytes: ByteArray, mime: String): ByteArray? {
        if (!wants()) return null
        try {
            val a = analyze(bytes) ?: return null
            if (a.confirmed.isEmpty()) return null
            val full = codec.full(bytes) ?: return null
            try {
                if (erase) Wm2Eraser.erase(full, a.confirmed, a.p, a.w2, a.h2)
                if (showBoxes) for (b in a.confirmed) drawBox(full, b)
                changed++
                return full.encode(mime.contains("png"))
            } finally { full.close() }
        } catch (t: Throwable) { lastErr = (t.javaClass.simpleName + ":" + (t.message ?: "")).take(80); return null }
    }

    private fun drawBox(f: WmFull, b: Wm2Detector.Box) {
        val red = 0xFFFF0000.toInt(); val t = if (b.tile) 8 else 4; val m = if (b.tile) 0 else 3
        val x0 = max(0, b.x - m); val y0 = max(0, b.y - m); val x1 = min(f.w, b.x + b.w + m); val y1 = min(f.h, b.y + b.h + m)
        fun fill(x: Int, y: Int, rw: Int, rh: Int) { if (rw <= 0 || rh <= 0) return; f.setRegion(x, y, rw, rh, IntArray(rw * rh) { red }) }
        fill(x0, y0, x1 - x0, min(t, y1 - y0)); fill(x0, max(y0, y1 - t), x1 - x0, min(t, y1 - y0))
        fill(x0, y0, min(t, x1 - x0), y1 - y0); fill(max(x0, x1 - t), y0, min(t, x1 - x0), y1 - y0)
    }

    companion object { const val MAX_EXAMPLES = 8 }
}
