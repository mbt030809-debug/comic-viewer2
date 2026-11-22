import com.example.comicviewer.wm2.*
import java.io.File

/** 인식 → 제거 전체 경로를 벤치에 돌려 지운 이미지를 PNG 로 저장한다(원본과의 비교는 eval_erase.py).
 *  사용: Wm2EraseBench <모델.onnx> <벤치폴더> <출력폴더> [사이트필터] */
fun main(args: Array<String>) {
    val model = File(args[0]).readBytes(); val root = File(args[1]); val outd = File(args[2]); val filt = args.getOrNull(3) ?: ""
    val det = Wm2Detector(model, 3); val t0 = System.currentTimeMillis(); var n = 0; var ch = 0
    for (s in root.listFiles { f -> f.isDirectory && f.name.contains(filt) }!!.sortedBy { it.name }) {
        val pipe = Wm2Pipeline(JvmCodec()); pipe.detector = det; pipe.erase = true; pipe.showBoxes = false
        val files = s.listFiles { f -> f.name.startsWith("img_") }!!.sortedBy { it.name }
        val bytes = files.map { it.readBytes() }
        for (b in bytes) pipe.process(b, "image/png")         // 1차: 앱이 받는 순서(확정 전 장은 원본)
        File(outd, s.name).mkdirs()
        for ((i, f) in files.withIndex()) {                    // 2차: 확정 뒤 다시 받기
            val r = pipe.process(bytes[i], "image/png"); n++
            if (r != null) { ch++; File(outd, s.name + "/" + f.nameWithoutExtension + ".png").writeBytes(r) }
        }
    }
    println("완료 ${n}장(바뀐 ${ch}장) ${(System.currentTimeMillis() - t0) / 1000}초 → $outd")
}
