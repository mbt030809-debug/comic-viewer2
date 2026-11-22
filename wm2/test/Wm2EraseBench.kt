import com.example.comicviewer.wm2.*
import java.io.File

/** 인식 → 제거 전체 경로를 벤치에 돌려 지운 이미지를 PNG 로 저장한다(원본과의 비교는 eval_erase.py).
 *  사용: Wm2EraseBench <모델.onnx> <벤치폴더> <출력폴더> [사이트필터] [erase=0|1,show=0|1]
 *  스위치 역할 시험: 결과에서 입력과 달라진 픽셀 수, 그중 순수 빨강(상자) 픽셀 수를 센다. */
fun main(args: Array<String>) {
    val model = File(args[0]).readBytes(); val root = File(args[1]); val outd = File(args[2]); val filt = args.getOrNull(3) ?: ""
    val flags = (args.getOrNull(4) ?: "erase=1,show=0").split(",").associate { it.substringBefore("=") to (it.substringAfter("=") == "1") }
    val det = Wm2Detector(model, 3); val t0 = System.currentTimeMillis(); var n = 0; var ch = 0; var diffPx = 0L; var redPx = 0L
    for (s in root.listFiles { f -> f.isDirectory && f.name.contains(filt) }!!.sortedBy { it.name }) {
        val pipe = Wm2Pipeline(JvmCodec()); pipe.detector = det; pipe.erase = flags["erase"] ?: true; pipe.showBoxes = flags["show"] ?: false
        val files = s.listFiles { f -> f.name.startsWith("img_") }!!.sortedBy { it.name }
        val bytes = files.map { it.readBytes() }
        for (b in bytes) pipe.process(b, "image/png")         // 1차: 앱이 받는 순서(확정 전 장은 원본)
        File(outd, s.name).mkdirs()
        for ((i, f) in files.withIndex()) {                    // 2차: 확정 뒤 다시 받기
            val r = pipe.process(bytes[i], "image/png"); n++
            if (r != null) {
                ch++; File(outd, s.name + "/" + f.nameWithoutExtension + ".png").writeBytes(r)
                val a = javax.imageio.ImageIO.read(bytes[i].inputStream()); val b = javax.imageio.ImageIO.read(r.inputStream())
                for (y in 0 until a.height) for (x in 0 until a.width) { val p = a.getRGB(x, y) and 0xFFFFFF; val q = b.getRGB(x, y) and 0xFFFFFF; if (p != q) { diffPx++; if (q == 0xFF0000) redPx++ } }
            }
        }
    }
    println("[지우기=${flags["erase"]} 표시=${flags["show"]}] ${n}장 중 바뀐 ${ch}장 · 입력과 달라진 픽셀 $diffPx 중 순수 빨강 $redPx · ${(System.currentTimeMillis() - t0) / 1000}초")
}
