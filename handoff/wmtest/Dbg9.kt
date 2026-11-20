import com.example.comicviewer.*
import java.io.File
import javax.imageio.ImageIO
fun main() {
    val dirIn = System.getProperty("real.in") ?: "/tmp/real_in"; val dirOut = System.getProperty("real.out") ?: "/tmp/real_out"
    File(dirOut).mkdirs()
    val e = mkEngine(JvmCodec(), null)
    val files = File(dirIn).listFiles()!!.filter { it.name.endsWith(".jpg") }.sortedBy { it.name }
    for ((k, f) in files.withIndex()) {
        val bytes = f.readBytes()
        val out = e.process(bytes, "image/jpeg")
        Thread.sleep((System.getProperty("wm.sleep") ?: "1500").toLong())
        File(dirOut, f.name).writeBytes(out ?: bytes)
        println(String.format("%2d %-24s %s | %s", k, f.name, if (out != null) "MARK" else "-   ", e.status()))
    }
    println(e.fullDiag())
}
