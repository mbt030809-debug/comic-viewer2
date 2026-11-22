package com.example.comicviewer.wm2

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.comicviewer.WmImg
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * 참조형 인식(assets/wm2_ref_unet.onnx). 입력 두 개:
 *  - image: 검사할 이미지를 0.5배(2x2 평균)로 줄인 RGB 0..1, 아래·오른쪽 reflect 패딩으로 16의 배수
 *  - ref  : 예시 마크(Wm2Gallery.makeRef) 112x192
 * 출력은 "그 예시 마크가 여기 박혀 있을 확률" 지도(0.5배 좌표). 예시마다 한 번씩 돌려 상자로 바꾼다.
 * 후처리는 학습 쪽 infer_ref.py 의 boxes_from_prob 와 같다(확률>th → 6칸 팽창 → 덩어리 → 30픽셀 미만 제거 → 좌표 ×2).
 * 상자 면적이 화면의 20% 를 넘으면 버린다(화면 전체를 워터마크로 잡는 것 방지).
 * 여러 장에서 반복되는지 확인하는 단계는 Wm2Confirmer 가 한다.
 */
class Wm2Detector(model: ByteArray, threads: Int = 2) {
    /** ref = 예시 마크 번호(갤러리 위→아래 순서). tile 은 예전 전면 무늬용 자리로 항상 false */
    class Box(val x: Int, val y: Int, val w: Int, val h: Int, val tile: Boolean, val score: Float, val ref: Int = 0)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val o = OrtSession.SessionOptions()
        o.setIntraOpNumThreads(threads)
        session = env.createSession(model, o)
    }

    /** 이미지 쪽 입력(예시와 무관하므로 한 번만 만들어 예시마다 재사용) */
    class Target(val data: FloatArray, val W: Int, val H: Int, val w2: Int, val h2: Int)

    fun prepare(img: WmImg): Target {
        val w2 = img.w / 2; val h2 = img.h / 2
        val pw = (16 - w2 % 16) % 16; val ph = (16 - h2 % 16) % 16
        val W = w2 + pw; val H = h2 + ph
        val half = FloatArray(3 * w2 * h2)
        val px = img.px; val iw = img.w
        for (y in 0 until h2) for (x in 0 until w2) {
            var r = 0; var g = 0; var b = 0
            for (dy in 0..1) { val o = (2 * y + dy) * iw + 2 * x; for (dx in 0..1) { val p = px[o + dx]; r += p shr 16 and 255; g += p shr 8 and 255; b += p and 255 } }
            val i = y * w2 + x
            half[i] = r / 1020f; half[w2 * h2 + i] = g / 1020f; half[2 * w2 * h2 + i] = b / 1020f
        }
        val inp = FloatArray(3 * H * W)
        for (c in 0 until 3) for (y in 0 until H) {
            val sy = reflect(y, h2); val row = c * w2 * h2 + sy * w2; val o = c * H * W + y * W
            for (x in 0 until W) inp[o + x] = half[row + reflect(x, w2)]
        }
        return Target(inp, W, H, w2, h2)
    }

    /** 이 예시 마크가 있을 확률 지도(0.5배 좌표, w2*h2) */
    fun probMap(t: Target, ref: FloatArray): FloatArray {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(t.data), longArrayOf(1, 3, t.H.toLong(), t.W.toLong())).use { ti ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(ref), longArrayOf(1, 3, Wm2Gallery.RH.toLong(), Wm2Gallery.RW.toLong())).use { tr ->
                session.run(mapOf("image" to ti, "ref" to tr)).use { r ->
                    val fb = (r[0] as OnnxTensor).floatBuffer
                    val lo = FloatArray(fb.remaining()); fb.get(lo)
                    val p = FloatArray(t.w2 * t.h2)
                    for (y in 0 until t.h2) for (x in 0 until t.w2) p[y * t.w2 + x] = 1f / (1f + exp(-lo[y * t.W + x]))
                    return p
                }
            }
        }
    }

    private fun reflect(i: Int, n: Int) = if (i < n) i else 2 * (n - 1) - i

    /** 상자 목록 + 확률 지도(예시들의 최댓값, 0.5배 좌표. 지우기가 마스크로 씀) */
    class Det(val boxes: List<Box>, val p: FloatArray, val w2: Int, val h2: Int)

    fun detect(img: WmImg, refs: List<FloatArray>, th: Float = 0.5f): Det {
        val t = prepare(img)
        val all = ArrayList<Box>(); val pmax = FloatArray(t.w2 * t.h2)
        for ((i, ref) in refs.withIndex()) {
            val p = probMap(t, ref)
            for (k in p.indices) if (p[k] > pmax[k]) pmax[k] = p[k]
            for (b in boxesFromMap(p, t.w2, t.h2, th, i)) if (b.w.toLong() * b.h <= MAX_AREA * img.w * img.h) all.add(b)
        }
        return Det(dedupe(all), pmax, t.w2, t.h2)
    }

    /** 서로 다른 예시가 같은 자리를 가리키면(비슷한 모양의 예시끼리 흔함) 점수 높은 상자 하나만 남긴다 */
    private fun dedupe(bs: List<Box>): List<Box> {
        val out = ArrayList<Box>()
        for (b in bs.sortedByDescending { it.score }) if (out.none { iou(it, b) >= 0.5f }) out.add(b)
        return out
    }
    private fun iou(a: Box, b: Box): Float {
        val ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x)); val iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        val i = ix.toLong() * iy; val u = a.w.toLong() * a.h + b.w.toLong() * b.h - i
        return if (u > 0) i.toFloat() / u else 0f
    }

    companion object {
        const val MERGE = 6          // 팽창 칸 수(0.5배 좌표)
        const val MIN_AREA = 30
        const val MAX_AREA = 0.20f   // 상자 면적 / 화면 면적 상한

        /** 확률 지도 → 상자(원본 좌표). ref 는 상자에 붙일 예시 번호 */
        fun boxesFromMap(p: FloatArray, w2: Int, h2: Int, th: Float, ref: Int): List<Box> {
            val n = w2 * h2; val m = BooleanArray(n); var cnt = 0
            for (i in 0 until n) if (p[i] > th) { m[i] = true; cnt++ }
            if (cnt == 0) return emptyList()
            // 마름모 팽창(4방향 MERGE번)
            var d = m.copyOf()
            repeat(MERGE) {
                val nd = d.copyOf()
                for (y in 0 until h2) for (x in 0 until w2) { val i = y * w2 + x; if (d[i]) { if (x > 0) nd[i - 1] = true; if (x < w2 - 1) nd[i + 1] = true; if (y > 0) nd[i - w2] = true; if (y < h2 - 1) nd[i + w2] = true } }
                d = nd
            }
            val lab = IntArray(n); val stack = IntArray(n); val out = ArrayList<Box>(); var nl = 0
            for (i0 in 0 until n) {
                if (!d[i0] || lab[i0] != 0) continue
                nl++; var sp = 0; stack[sp++] = i0; lab[i0] = nl
                var x0 = Int.MAX_VALUE; var y0 = Int.MAX_VALUE; var x1 = -1; var y1 = -1; var c = 0; var s = 0f
                while (sp > 0) {
                    val i = stack[--sp]; val x = i % w2; val y = i / w2
                    if (m[i]) { c++; s += p[i]; x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y) }
                    if (x > 0 && d[i - 1] && lab[i - 1] == 0) { lab[i - 1] = nl; stack[sp++] = i - 1 }
                    if (x < w2 - 1 && d[i + 1] && lab[i + 1] == 0) { lab[i + 1] = nl; stack[sp++] = i + 1 }
                    if (y > 0 && d[i - w2] && lab[i - w2] == 0) { lab[i - w2] = nl; stack[sp++] = i - w2 }
                    if (y < h2 - 1 && d[i + w2] && lab[i + w2] == 0) { lab[i + w2] = nl; stack[sp++] = i + w2 }
                }
                if (c < MIN_AREA || x1 < 0) continue
                out.add(Box(x0 * 2, y0 * 2, (x1 - x0 + 1) * 2, (y1 - y0 + 1) * 2, false, s / c, ref))
            }
            return out
        }
    }
}
