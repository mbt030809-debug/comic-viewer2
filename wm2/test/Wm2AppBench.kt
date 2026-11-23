import com.example.comicviewer.wm2.*
import java.io.File

/** 앱과 같은 Kotlin 경로(Wm2Gallery → Wm2Detector → Wm2Confirmer)를 JVM 에서 참조형 벤치(세트 폴더: gallery.png + img_NN)에 돌려 인식만 평가한다.
 *  사용: Wm2AppBench <2단계모델.onnx> <벤치폴더> <출력.tsv> [세트필터] [1단계모델.onnx]
 *  앱처럼 갤러리를 먼저 넣고 이미지를 차례로 처리한 뒤(1차), 확정 후 앞선 장을 다시 받는 것(2차)까지 흉내 내 2차 결과를 기록한다.
 *  출력 줄: 세트 \t 파일 \t 예시번호 \t x \t y \t w \t h \t 점수   (상자 없으면 NONE) */
fun main(args: Array<String>) {
    val model = File(args[0]).readBytes(); val root = File(args[1]); val out = File(args[2]); val filt = args.getOrNull(3) ?: ""
    val s1 = args.getOrNull(4)?.let { Wm2Stage1(File(it).readBytes(), 3) }   // 1단계 신경망(여백에서 예시 찾기). 없으면 갤러리 이미지만 예시로 씀
    val det = Wm2Detector(model, 3); val sb = StringBuilder(); val t0 = System.currentTimeMillis(); var n = 0
    for (s in root.listFiles { f -> f.isDirectory && f.name.contains(filt) && (File(f, "gallery.png").exists() || s1 != null) }!!.sortedBy { it.name }) {
        val pipe = Wm2Pipeline(JvmCodec()); pipe.detector = det; pipe.stage1 = s1
        if (File(s, "gallery.png").exists()) pipe.analyze(File(s, "gallery.png").readBytes())
        sb.append("${s.name}\tgallery\tREFS\t${pipe.refCount}\n")
        val files = s.listFiles { f -> f.name.startsWith("img_") }!!.sortedBy { it.name }
        val bytes = files.map { it.readBytes() }
        for (b in bytes) pipe.analyze(b)
        for ((i, f) in files.withIndex()) {
            val r = pipe.analyze(bytes[i])?.confirmed; n++
            if (r == null || r.isEmpty()) sb.append("${s.name}\t${f.name}\tNONE\n") else for (b in r) sb.append("${s.name}\t${f.name}\t${b.ref}\t${b.x}\t${b.y}\t${b.w}\t${b.h}\t${"%.3f".format(b.score)}\n")
        }
    }
    out.writeText(sb.toString()); println("완료 ${n}장 ${(System.currentTimeMillis() - t0) / 1000}초 → $out")
}
