package com.example.comicviewer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/** ONNX Runtime 으로 워터마크 제거 신경망(assets/wm_net.onnx)을 돌림. 라이브러리/모델이 없거나 실패하면 만들 때 예외 → 호출한 쪽이 null 로 두고 계산식만 사용 */
class OrtNet(model: ByteArray) : WmNet {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val o = OrtSession.SessionOptions()
        o.setIntraOpNumThreads(2)
        session = env.createSession(model, o)
    }

    override fun infer(x: FloatArray, h: Int, w: Int): FloatArray? {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 9, h.toLong(), w.toLong())).use { t ->
            session.run(mapOf("x" to t)).use { r ->
                val fb = (r[0] as OnnxTensor).floatBuffer
                val out = FloatArray(fb.remaining()); fb.get(out)
                return out
            }
        }
    }
}
