import com.example.comicviewer.*
import java.io.File
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
fun main() { for (nm in listOf("bbeba96e","e95a9e8c","26f7ad58")) one(nm) }
fun one(nm: String) {
    val bytes = File("/root/.claude/uploads/0fcee87f-932b-528f-b286-2b28533fd095/$nm-image.jpg").readBytes()
    val c = JvmCodec(); val dim = c.info(bytes)!!
    val s = WmCore.scaleFor(dim[0]); println("w=${dim[0]} h=${dim[1]} s=$s")
    val small = c.small(bytes, s)!!
    val th = WmCore.makeThumb(small, s, dim[0], dim[1])
    println("calm=${th.calm} tw=${th.tw} th=${th.th}")
    for (name in listOf("g")) {
        val img = BufferedImage(th.tw, th.th, BufferedImage.TYPE_INT_RGB)
        for (y in 0 until th.th) for (x in 0 until th.tw) { val v = Math.min(255, (th.g[y * th.tw + x] * 4).toInt()); img.setRGB(x, y, (v shl 16) or (v shl 8) or v) }
        ImageIO.write(img, "png", File("/tmp/g_$nm.png"))
    }
    // 가장 윤곽이 강하게 남은 가로 줄
    val rows = (0 until th.th).map { y -> var sum = 0f; for (x in 0 until th.tw) sum += th.g[y * th.tw + x]; y to sum }.sortedByDescending { it.second }.take(5)
    println("강한 줄: " + rows.joinToString { "y=${it.first * s} sum=${"%.0f".format(it.second)}" })
}
