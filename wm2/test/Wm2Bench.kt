import com.example.comicviewer.wm2.*
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.imageio.ImageIO

/** 사용: Wm2Bench <벤치폴더(core/wild/val)> <예측출력파일> [사이트이름부분문자열] [--shuffle=seed] [--subset=n]
 *  사이트마다 이미지(img_*)를 읽어 Wm2.recognize 에 넣고, 예측을 탭 구분으로 적는다. */
fun load(f: File): Img {
    val bi = ImageIO.read(f); val w = bi.width; val h = bi.height; val px = IntArray(w * h)
    bi.getRGB(0, 0, w, h, px, 0, w); return Img(w, h, px)
}

fun main(args: Array<String>) {
    val root = File(args[0]); val out = File(args[1]); val filt = args.getOrNull(2)?.takeIf { !it.startsWith("--") } ?: ""
    val shuffle = args.firstOrNull { it.startsWith("--shuffle=") }?.substringAfter("=")?.toLong()
    val subset = args.firstOrNull { it.startsWith("--subset=") }?.substringAfter("=")?.toInt()
    val sites = root.listFiles { f -> f.isDirectory && f.name.contains(filt) }!!.sortedBy { it.name }
    val ex = Executors.newFixedThreadPool(System.getProperty("jobs")?.toInt() ?: 3)
    val results = java.util.concurrent.ConcurrentHashMap<String, String>()
    val t0 = System.currentTimeMillis()
    for (s in sites) ex.submit {
        try {
            var files = s.listFiles { f -> f.name.startsWith("img_") }!!.sortedBy { it.name }
            if (shuffle != null) files = files.shuffled(java.util.Random(shuffle + s.name.hashCode()))
            if (subset != null) files = files.take(subset)
            val imgs = files.map { load(it) }
            val t1 = System.currentTimeMillis()
            val r = Wm2.recognize(imgs)
            val sb = StringBuilder()
            for ((i, f) in files.withIndex()) {
                if (r.marks[i].isEmpty()) sb.append("${s.name}\t${f.name}\tNONE\n")
                for (m in r.marks[i]) sb.append("${s.name}\t${f.name}\t${m.x}\t${m.y}\t${m.w}\t${m.h}\t${m.kind}\t${"%.3f".format(m.score)}\n")
            }
            for (l in r.log) { if (l.startsWith("@cand\t") || l.startsWith("@cl\t")) { val p = l.split("\t").toMutableList(); p[1] = files[p[1].toInt()].name; sb.append("@${s.name}\t${p.drop(1).joinToString("\t")}\t${p[0]}\n") } else sb.append("#${s.name}\t$l\n") }
            sb.append("#${s.name}\t시간 ${System.currentTimeMillis() - t1}ms\n")
            results[s.name] = sb.toString()
        } catch (e: Throwable) { results[s.name] = "#${s.name}\t오류 $e\n"; e.printStackTrace() }
    }
    ex.shutdown(); ex.awaitTermination(3, TimeUnit.HOURS)
    out.writeText(sites.joinToString("") { results[it.name] ?: "" })
    println("완료 ${sites.size}개 사이트, ${(System.currentTimeMillis() - t0) / 1000}초 → $out")
}
