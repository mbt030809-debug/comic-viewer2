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
 * 학습된 U-Net(ONNX, assets/wm2_unet.onnx)으로 이미지 한 장에서 워터마크 후보 상자를 찾는다.
 * 입력: 원본을 0.5배(2x2 평균)로 줄인 RGB 0..1, 크기는 16의 배수가 되도록 아래·오른쪽 reflect 패딩. 출력: 워터마크 로짓 지도.
 * 후처리는 학습 쪽 infer_eval.py 의 boxes_from_mask 와 같다(확률>th → 12% 초과면 전면 무늬, 아니면 6칸 팽창 후 덩어리 상자).
 * 여러 장에서 반복되는지 확인하는 단계는 Wm2Engine 이 한다.
 */
class Wm2Detector(model: ByteArray, threads: Int = 2) {
    class Box(val x: Int, val y: Int, val w: Int, val h: Int, val tile: Boolean, val score: Float)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val o = OrtSession.SessionOptions()
        o.setIntraOpNumThreads(threads)
        session = env.createSession(model, o)
    }

    /** 확률 지도(0.5배 축소 좌표, 크기 w2*h2)를 돌려준다. 실패하면 예외 */
    fun probMap(img: WmImg): Triple<FloatArray, Int, Int> {
        val w2 = img.w / 2; val h2 = img.h / 2
        val pw = (16 - w2 % 16) % 16; val ph = (16 - h2 % 16) % 16
        val W = w2 + pw; val H = h2 + ph
        // 0.5배 평균(RGB 0..1)
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
        OnnxTensor.createTensor(env, FloatBuffer.wrap(inp), longArrayOf(1, 3, H.toLong(), W.toLong())).use { t ->
            session.run(mapOf("image" to t)).use { r ->
                val fb = (r[0] as OnnxTensor).floatBuffer
                val lo = FloatArray(fb.remaining()); fb.get(lo)
                val p = FloatArray(w2 * h2)
                for (y in 0 until h2) for (x in 0 until w2) p[y * w2 + x] = 1f / (1f + exp(-lo[y * W + x]))
                return Triple(p, w2, h2)
            }
        }
    }

    private fun reflect(i: Int, n: Int) = if (i < n) i else 2 * (n - 1) - i

    fun detect(img: WmImg, th: Float = 0.5f): List<Box> = detectMap(img, th).boxes

    class Det(val boxes: List<Box>, val p: FloatArray, val w2: Int, val h2: Int)
    /** 상자와 확률 지도(0.5배 좌표)를 함께 돌려준다(지우기는 지도를 마스크로 쓴다) */
    fun detectMap(img: WmImg, th: Float = 0.5f): Det {
        val (p, w2, h2) = probMap(img)
        return Det(boxesFromMap(p, w2, h2, img.w, img.h, th), p, w2, h2)
    }

    companion object {
        const val TILE_FRAC = 0.12f
        const val MERGE = 6          // 팽창 칸 수(0.5배 좌표)
        const val MIN_AREA = 30

        fun boxesFromMap(p: FloatArray, w2: Int, h2: Int, w: Int, h: Int, th: Float): List<Box> {
            val n = w2 * h2; val m = BooleanArray(n); var cnt = 0; var sum = 0f
            for (i in 0 until n) if (p[i] > th) { m[i] = true; cnt++; sum += p[i] }
            if (cnt.toFloat() / n > TILE_FRAC) return listOf(Box(0, 0, w, h, true, sum / cnt))
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
                out.add(Box(x0 * 2, y0 * 2, (x1 - x0 + 1) * 2, (y1 - y0 + 1) * 2, false, s / c))
            }
            return out
        }
    }
}
