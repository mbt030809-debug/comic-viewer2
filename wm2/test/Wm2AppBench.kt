import com.example.comicviewer.wm2.*
import java.io.File

/** 앱과 같은 Kotlin 경로(Wm2Detector → Wm2Confirmer)를 JVM 에서 벤치에 돌려 인식만 평가한다.
 *  사용: Wm2AppBench <모델.onnx> <벤치폴더> <출력.tsv> [사이트필터]
 *  앱처럼 장을 차례로 처리한 뒤(1차), 확정 후 앞선 장을 다시 받는 것(2차)까지 흉내 내 2차 결과를 기록한다. */
fun main(args: Array<String>) {
    val model = File(args[0]).readBytes(); val root = File(args[1]); val out = File(args[2]); val filt = args.getOrNull(3) ?: ""
    val det = Wm2Detector(model, 3); val sb = StringBuilder(); val t0 = System.currentTimeMillis(); var n = 0
    for (s in root.listFiles { f -> f.isDirectory && f.name.contains(filt) }!!.sortedBy { it.name }) {
        val pipe = Wm2Pipeline(JvmCodec()); pipe.detector = det
        val files = s.listFiles { f -> f.name.startsWith("img_") }!!.sortedBy { it.name }
        val bytes = files.map { it.readBytes() }
        for (b in bytes) pipe.analyze(b)
        for ((i, f) in files.withIndex()) {
            val r = pipe.analyze(bytes[i])?.confirmed; n++
            if (r == null || r.isEmpty()) sb.append("${s.name}\t${f.name}\tNONE\n") else for (b in r) sb.append("${s.name}\t${f.name}\t${b.x}\t${b.y}\t${b.w}\t${b.h}\t${if (b.tile) "tile" else "mark"}\t${"%.3f".format(b.score)}\n")
        }
    }
    out.writeText(sb.toString()); println("완료 ${n}장 ${(System.currentTimeMillis() - t0) / 1000}초 → $out")
}
