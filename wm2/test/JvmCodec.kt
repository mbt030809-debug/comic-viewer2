import com.example.comicviewer.*
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

/** PC/JVM 시험용 디코더(앱의 AndroidCodec 대응) */
class JvmFull(val bi: BufferedImage) : WmFull {
    override val w get() = bi.width
    override val h get() = bi.height
    override fun getRegion(x: Int, y: Int, rw: Int, rh: Int, out: IntArray) { bi.getRGB(x, y, rw, rh, out, 0, rw) }
    override fun setRegion(x: Int, y: Int, rw: Int, rh: Int, px: IntArray) { bi.setRGB(x, y, rw, rh, px, 0, rw) }
    override fun encode(png: Boolean): ByteArray { val o = ByteArrayOutputStream(); ImageIO.write(bi, "png", o); return o.toByteArray() }
    override fun close() {}
}

class JvmCodec : WmCodec {
    private fun rgb(b: ByteArray): BufferedImage? { val s = ImageIO.read(b.inputStream()) ?: return null; val o = BufferedImage(s.width, s.height, BufferedImage.TYPE_INT_RGB); o.createGraphics().apply { drawImage(s, 0, 0, null); dispose() }; return o }
    override fun info(bytes: ByteArray): IntArray? = ImageIO.read(bytes.inputStream())?.let { intArrayOf(it.width, it.height) }
    override fun small(bytes: ByteArray, s: Int): WmImg? {
        val bi = rgb(bytes) ?: return null; val w = bi.width / s; val h = bi.height / s; val px = IntArray(w * h)
        if (s == 1) bi.getRGB(0, 0, w, h, px, 0, w) else for (y in 0 until h) for (x in 0 until w) px[y * w + x] = bi.getRGB(x * s, y * s)
        return WmImg(w, h, px)
    }
    override fun full(bytes: ByteArray): WmFull? = rgb(bytes)?.let { JvmFull(it) }
}
