package com.example.comicviewer.wm2

import com.example.comicviewer.WmCodec
import com.example.comicviewer.WmFull
import kotlin.math.max
import kotlin.math.min

/**
 * 워터마크 처리의 전체 흐름. 이미지 한 장마다 한 방향으로만 흐른다:
 *   디코드 → [인식] Wm2Detector → [확정] Wm2Confirmer → [제거] Wm2Eraser → (표시) 빨간 상자 → 인코드
 * 지운 뒤 다시 인식하는 단계는 없다. 각 단계는 서로의 내부를 모른다.
 */
class Wm2Pipeline(private val codec: WmCodec) {
    class Analysis(val confirmed: List<Wm2Detector.Box>, val p: FloatArray, val w2: Int, val h2: Int)

    @Volatile var detector: Wm2Detector? = null      // 모델이 준비되기 전에는 아무것도 하지 않음
    @Volatile var erase = true                       // 확정된 워터마크 지우기
    @Volatile var showBoxes = false                  // 확정된 자리에 빨간 상자 표시
    @Volatile var logSink: ((String) -> Unit)? = null
    @Volatile var onConfirmed: (() -> Unit)? = null  // 처음 확정되는 순간: 앞서 원본으로 나간 이미지를 다시 받게 함
    @Volatile var lastErr = ""
    @Volatile private var seen = 0
    @Volatile private var changed = 0
    @Volatile private var lastMs = 0L
    private val confirmer = Wm2Confirmer()
    private val log = ArrayList<String>()

    fun wants(): Boolean = detector != null && (erase || showBoxes)
    fun onPage() = confirmer.reset()
    fun forget() = confirmer.reset()

    private fun note(t: String) { synchronized(log) { log.add(t); if (log.size > 60) log.removeAt(0) }; try { logSink?.invoke(t) } catch (_: Throwable) { } }
    fun status(): String =
        if (detector == null) "모델 준비 중"
        else if (!erase && !showBoxes) "꺼짐"
        else if (confirmer.everConfirmed) "워터마크 확정 · 처리 ${changed}장 · 검사 ${seen}장 · ${lastMs}ms/장"
        else "찾는 중 · 검사 ${seen}장(같은 너비 ${confirmer.size}장, 3장 이상이면 판정) · ${lastMs}ms/장"
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
        val d = det.detectMap(img)
        val boxes = Wm2Confirmer.tileMerge(d.boxes, w, h)
        lastMs = System.currentTimeMillis() - t0; seen++
        val r = confirmer.confirm(bytes.size * 31L + bytes.contentHashCode(), w, h, boxes)
        note("#$seen ${w}x$h 후보 ${boxes.size}개 · 비교 ${r.others}장" + (if (r.others < Wm2Confirmer.MIN_OTHERS) " → 판정 대기" else " → 확정 ${r.confirmed.size}개"))
        if (r.firstTime) try { onConfirmed?.invoke() } catch (_: Throwable) { }
        return Analysis(r.confirmed, d.p, d.w2, d.h2)
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
}
