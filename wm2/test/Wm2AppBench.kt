import com.example.comicviewer.*
import com.example.comicviewer.wm2.*
import java.io.File
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

/** 앱과 같은 Kotlin 경로(Wm2Detector+Wm2Engine)를 JVM 에서 벤치에 돌린다.
 *  사용: Wm2AppBench <모델.onnx> <벤치폴더> <출력.tsv> [사이트필터]
 *  앱처럼 장을 차례로 처리한 뒤(1차), 확정 후 앞선 장을 다시 받는 것(2차, retro)까지 흉내 내 2차 결과를 기록한다. */
class JvmCodec : WmCodec {
    override fun info(bytes: ByteArray): IntArray? = ImageIO.read(bytes.inputStream())?.let { intArrayOf(it.width, it.height) }
    override fun small(bytes: ByteArray, s: Int): WmImg? {
        val bi = ImageIO.read(bytes.inputStream()) ?: return null; val w = bi.width / s; val h = bi.height / s; val px = IntArray(w * h)
        if (s == 1) bi.getRGB(0, 0, w, h, px, 0, w) else for (y in 0 until h) for (x in 0 until w) px[y * w + x] = bi.getRGB(x * s, y * s)
        return WmImg(w, h, px)
    }
    override fun full(bytes: ByteArray): WmFull? = null
}

fun main(args: Array<String>) {
    val model = File(args[0]).readBytes(); val root = File(args[1]); val out = File(args[2]); val filt = args.getOrNull(3) ?: ""
    val det = Wm2Detector(model, 3); val sb = StringBuilder(); val t0 = System.currentTimeMillis(); var n = 0
    for (s in root.listFiles { f -> f.isDirectory && f.name.contains(filt) }!!.sortedBy { it.name }) {
        val eng = Wm2Engine(JvmCodec(), det)
        val files = s.listFiles { f -> f.name.startsWith("img_") }!!.sortedBy { it.name }
        val bytes = files.map { it.readBytes() }
        for (b in bytes) eng.analyze(b)                      // 1차: 앱이 이미지를 받는 순서
        for ((i, f) in files.withIndex()) {                   // 2차: 확정 뒤 다시 받기
            val r = eng.analyze(bytes[i]); n++
            if (r == null || r.isEmpty()) sb.append("${s.name}\t${f.name}\tNONE\n") else for (b in r) sb.append("${s.name}\t${f.name}\t${b.x}\t${b.y}\t${b.w}\t${b.h}\t${if (b.tile) "tile" else "mark"}\t${"%.3f".format(b.score)}\n")
        }
    }
    out.writeText(sb.toString()); println("완료 ${n}장 ${(System.currentTimeMillis() - t0) / 1000}초 → $out")
}
