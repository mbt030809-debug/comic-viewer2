import com.example.comicviewer.*
import java.awt.image.BufferedImage
import java.io.*
import javax.imageio.*
import javax.imageio.stream.MemoryCacheImageOutputStream
import kotlin.math.*

fun toRgb(bi: BufferedImage): BufferedImage { val o = BufferedImage(bi.width, bi.height, BufferedImage.TYPE_INT_RGB); val g = o.createGraphics(); g.drawImage(bi, 0, 0, null); g.dispose(); return o }
fun jpeg(bi: BufferedImage, q: Float): ByteArray {
    val w = ImageIO.getImageWritersByFormatName("jpeg").next(); val p = w.defaultWriteParam
    p.compressionMode = ImageWriteParam.MODE_EXPLICIT; p.compressionQuality = q
    val bo = ByteArrayOutputStream(); val ios = MemoryCacheImageOutputStream(bo); w.output = ios
    w.write(null, IIOImage(toRgb(bi), null, null), p); ios.close(); w.dispose(); return bo.toByteArray()
}
class JvmFull(val img: BufferedImage) : WmFull {
    override val w get() = img.width
    override val h get() = img.height
    override fun getRegion(x: Int, y: Int, rw: Int, rh: Int, out: IntArray) { img.getRGB(x, y, rw, rh, out, 0, rw) }
    override fun setRegion(x: Int, y: Int, rw: Int, rh: Int, px: IntArray) { img.setRGB(x, y, rw, rh, px, 0, rw) }
    override fun encode(png: Boolean): ByteArray { if (png) { val bo = ByteArrayOutputStream(); ImageIO.write(img, "png", bo); return bo.toByteArray() }; return jpeg(img, 0.95f) }
    override fun close() {}
}
class JvmCodec : WmCodec {
    override fun info(bytes: ByteArray): IntArray? {
        val iis = ImageIO.createImageInputStream(ByteArrayInputStream(bytes)); val it = ImageIO.getImageReaders(iis); if (!it.hasNext()) return null
        val r = it.next(); r.input = iis; val a = intArrayOf(r.getWidth(0), r.getHeight(0)); r.dispose(); return a
    }
    override fun small(bytes: ByteArray, s: Int): WmImg? {
        val bi = ImageIO.read(ByteArrayInputStream(bytes)) ?: return null
        val sw = (bi.width + s - 1) / s; val sh = (bi.height + s - 1) / s
        val src = IntArray(bi.width * bi.height); bi.getRGB(0, 0, bi.width, bi.height, src, 0, bi.width)
        val o = IntArray(sw * sh)
        for (y in 0 until sh) for (x in 0 until sw) {
            var r = 0; var g = 0; var b = 0; var c = 0
            for (yy in y * s until min(bi.height, y * s + s)) for (xx in x * s until min(bi.width, x * s + s)) { val p = src[yy * bi.width + xx]; r += (p shr 16) and 255; g += (p shr 8) and 255; b += p and 255; c++ }
            o[y * sw + x] = (255 shl 24) or ((r / c) shl 16) or ((g / c) shl 8) or (b / c)
        }
        return WmImg(sw, sh, o)
    }
    override fun full(bytes: ByteArray): WmFull? {
        val bi = ImageIO.read(ByteArrayInputStream(bytes)) ?: return null
        val a = BufferedImage(bi.width, bi.height, BufferedImage.TYPE_INT_ARGB); val g = a.createGraphics(); g.drawImage(bi, 0, 0, null); g.dispose()
        return JvmFull(a)
    }
}
class Stamp(val w: Int, val h: Int, val u: FloatArray, val l: FloatArray, val lc: Array<FloatArray>? = null)
val A = "assets/"
fun gray(path: String): Pair<IntArray, IntArray> { val bi = ImageIO.read(File(path)); val o = IntArray(bi.width * bi.height); for (y in 0 until bi.height) for (x in 0 until bi.width) o[y * bi.width + x] = bi.raster.getSample(x, y, 0); return o to intArrayOf(bi.width, bi.height) }
fun loadStamp(a: String, b: String): Stamp { val (u, d) = gray(A + a); val (l, _) = gray(A + b); return Stamp(d[0], d[1], FloatArray(u.size) { u[it] / 255f }, FloatArray(l.size) { l[it].toFloat() }) }
val W = 1248
fun flat(h: Int, r: Int, g: Int, b: Int) = IntArray(W * h) { (255 shl 24) or (r shl 16) or (g shl 8) or b }
val artCache = HashMap<Int, IntArray>()
fun art(i: Int, h: Int, off: Int = 0): IntArray {
    val a = artCache.getOrPut(i) { val bi = ImageIO.read(File(A + "art$i.png")); IntArray(bi.width * bi.height).also { bi.getRGB(0, 0, bi.width, bi.height, it, 0, bi.width) } }
    val ah = a.size / W; val o = IntArray(W * h); for (y in 0 until h) System.arraycopy(a, ((y + off) % ah) * W, o, y * W, W); return o
}
fun stampOn(j: IntArray, h: Int, st: Stamp, x0: Int, y0: Int): IntArray {
    val o = j.copyOf()
    for (y in 0 until st.h) for (x in 0 until st.w) {
        val a = st.u[y * st.w + x]; if (a <= 0f) continue
        val i = (y0 + y) * W + x0 + x; val p = j[i]; val l = st.l[y * st.w + x]
        val lr = st.lc?.get(0)?.get(y * st.w + x) ?: l; val lg = st.lc?.get(1)?.get(y * st.w + x) ?: l; val lb = st.lc?.get(2)?.get(y * st.w + x) ?: l
        val r = (((p shr 16) and 255) * (1 - a) + a * lr).roundToInt(); val g = (((p shr 8) and 255) * (1 - a) + a * lg).roundToInt(); val b = ((p and 255) * (1 - a) + a * lb).roundToInt()
        o[i] = (255 shl 24) or (r.coerceIn(0, 255) shl 16) or (g.coerceIn(0, 255) shl 8) or b.coerceIn(0, 255)
    }
    return o
}
fun bi(px: IntArray, h: Int): BufferedImage { val b = BufferedImage(W, h, BufferedImage.TYPE_INT_ARGB); b.setRGB(0, 0, W, h, px, 0, W); return b }
fun decode(bytes: ByteArray): IntArray { val b = ImageIO.read(ByteArrayInputStream(bytes)); val o = IntArray(b.width * b.height); b.getRGB(0, 0, b.width, b.height, o, 0, b.width); return o }
fun psnr(a: IntArray, b: IntArray, x0: Int, y0: Int, rw: Int, rh: Int): Double {
    var se = 0.0
    for (y in y0 until y0 + rh) for (x in x0 until x0 + rw) { val p = a[y * W + x]; val q = b[y * W + x]
        for (s in intArrayOf(16, 8, 0)) { val d = ((p shr s) and 255) - ((q shr s) and 255); se += d * d } }
    val mse = se / (rw * rh * 3); return if (mse < 1e-9) 99.0 else 10 * log10(255.0 * 255.0 / mse)
}
/** 그림이 서로 다른 시험 배경: 그림마다 큰 세로 이동 + 가끔 좌우 뒤집기 + 밝기 변화(같은 그림이 반복돼 생기는 가짜 반복을 줄임) */
fun bg2(name: String, h: Int, k: Int): IntArray {
    if (!name.startsWith("art")) return bg(name, h, 0)
    val o = art(name.substring(3).toInt(), h, (k * 211 + 97) % (1490 - 100))
    val flip = (k / 7) % 2 == 1; val mul = 0.75f + 0.08f * (k % 4)
    val r = IntArray(o.size)
    for (y in 0 until h) for (x in 0 until W) { val p = o[y * W + (if (flip) W - 1 - x else x)]
        val rr = (((p shr 16) and 255) * mul).toInt().coerceIn(0, 255); val gg = (((p shr 8) and 255) * mul).toInt().coerceIn(0, 255); val bb = ((p and 255) * mul).toInt().coerceIn(0, 255)
        r[y * W + x] = (255 shl 24) or (rr shl 16) or (gg shl 8) or bb }
    return r
}
fun bg(name: String, h: Int, off: Int = 0): IntArray = when {
    name == "white" -> flat(h, 255, 255, 255); name == "black" -> flat(h, 8, 8, 12); name == "navy" -> flat(h, 22, 30, 70)
    name == "beige" -> flat(h, 238, 226, 200); name == "gray" -> flat(h, 128, 128, 128)
    name.startsWith("art") -> art(name.substring(3).toInt(), h, off)
    else -> error(name)
}
fun savePng(px: IntArray, h: Int, path: String, x0: Int, y0: Int, rw: Int, rh: Int) { val o = BufferedImage(rw, rh, BufferedImage.TYPE_INT_RGB); for (y in 0 until rh) for (x in 0 until rw) o.setRGB(x, y, px[(y0 + y) * W + x0 + x]); ImageIO.write(o, "png", File(path)) }

fun runSeq(title: String, engine: WmEngine, names: List<String>, st: Stamp?, h: Int, sx0: Int, sy: Int, bottomDist: Int = -1, heights: List<Int>? = null, show: Boolean = true, frac: Double = -1.0, rnd: Boolean = false, sxs: List<Int>? = null, deco: ((IntArray, Int, Int) -> Unit)? = null) {
    println("=== $title ==="); val codec = JvmCodec()
    for ((k, n) in names.withIndex()) {
        val sx = sxs?.get(k) ?: sx0
        val hh = heights?.get(k) ?: h
        val j = if (System.getProperty("wm.bg2") != null) bg2(n, hh, k) else bg(n, hh, k * 37)
        val y0 = if (rnd && st != null) (80 + (k * 197 + 53) % (hh - st.h - 160)) else if (frac >= 0 && st != null) Math.round(frac * (hh - st.h)).toInt() else if (bottomDist >= 0 && st != null) hh - bottomDist - st.h else sy
        deco?.invoke(j, hh, k)
        val i = if (st != null) stampOn(j, hh, st, sx, y0) else j
        val bytes = jpeg(bi(i, hh), 0.92f)
        if (System.getenv("WMDBG") != null) System.err.println("[slice $k $n] h=$hh stamp x=$sx y=$y0")
        val t0 = System.nanoTime(); val out = engine.process(bytes, "image/jpeg"); val ms = (System.nanoTime() - t0) / 1e6
        Thread.sleep((System.getProperty("wm.sleep") ?: "250").toLong()) // 비동기 감지 대기
        System.getProperty("wm.outdir")?.let { d -> java.io.File(d, title.take(9).replace(" ", "_").replace(":", "") + "_" + k + "_" + (if (out != null) "C" else "N") + ".bin").writeBytes(out ?: bytes) }
        val inPx = decode(bytes); val outPx = if (out != null) decode(out) else inPx
        val rw = st?.w ?: 280; val rh = st?.h ?: 314
        val pIn = psnr(inPx, j, sx, y0, rw, rh); val pOut = psnr(outPx, j, sx, y0, rw, rh)
        System.getProperty("wm.dump")?.let { dir -> if (out != null && st != null) {
            val cw = rw + 20; val ch = rh + 20; val img = java.awt.image.BufferedImage(cw * 3, ch, java.awt.image.BufferedImage.TYPE_INT_RGB)
            for (yy in 0 until ch) for (xx in 0 until cw) { val gx = sx - 10 + xx; val gy = y0 - 10 + yy
                if (gx in 0 until W && gy in 0 until hh) { val q = gy * W + gx; img.setRGB(xx, yy, inPx[q] and 0xFFFFFF); img.setRGB(cw + xx, yy, outPx[q] and 0xFFFFFF); img.setRGB(2 * cw + xx, yy, j[q] and 0xFFFFFF) } }
            javax.imageio.ImageIO.write(img, "png", File(dir, "s_" + title.take(1) + "_" + k + "_" + n + ".png")) } }
        var dotInfo = ""
        if (out != null && st != null) { var sxr = 0L; var syr = 0L; var cnt = 0
            for (q in outPx.indices) { val p = outPx[q]; if (((p shr 16) and 255) in 214..234 && ((p shr 8) and 255) < 14 && (p and 255) < 14) { sxr += q % W; syr += q / W; cnt++ } }
            if (cnt > 60) { val dx = (sxr / cnt) - (sx + st.w / 2); val dy = (syr / cnt) - (y0 + st.h / 2); dotInfo = " dot(dx=$dx,dy=$dy)" } }
        if (show) println(String.format("%2d %-6s h=%d  in=%.1fdB out=%.1fdB %s %5.0fms | %s%s", k, n, hh, pIn, pOut, if (out != null) "CLEANED" else "-      ", ms, engine.status(), dotInfo))
    }
}


// 컷 테두리·말풍선 같은 반복 프레임(불투명)을 그림 위에 그림: 가로 위치는 모든 이미지에서 같고 세로 위치는 제각각
fun frameDeco(j: IntArray, hh: Int, k: Int) {
    fun rect(x0: Int, y0: Int, x1: Int, y1: Int, t: Int, c: Int) {
        for (y in y0..y1) for (x in x0..x1) { if (x !in 0 until W || y !in 0 until hh) continue
            val edge = x < x0 + t || x > x1 - t || y < y0 + t || y > y1 - t
            if (edge) j[y * W + x] = c }
    }
    val a = 40 + (k * 97) % 200; val b = hh - 40 - (k * 61) % 200
    rect(18, a, W - 18, b, 4, 0xFF101010.toInt())              // 큰 컷 테두리(가로 위치 고정)
    rect(18, a + 200, W / 2 - 6, b - 150, 3, 0xFF181818.toInt()) // 안쪽 컷 구분선
    rect(W - 260, a + 60, W - 60, a + 170, 3, 0xFF000000.toInt()) // 말풍선 비슷한 상자
    // 기울어진 굵은 선(컷 사선 경계): 기울기와 높이가 이미지마다 다름
    val y0 = 120 + (k * 53) % 300; val slope = ((k * 7) % 11 - 5) / 20.0
    for (x in 0 until W) { val yc = (y0 + slope * x).toInt(); for (t in 0 until 7) { val y = yc + t; if (y in 0 until hh) j[y * W + x] = 0xFF050505.toInt() } }
}


// 사용자 스크린샷의 붉은 원형 도장 무늬(배경을 흐린 것과의 차이)를 뽑아 다른 그림 위에 얹음
object RedStamp {
    const val X0 = 840; const val Y0 = 1030; const val CW = 390; const val CH = 360
    val hp: Array<FloatArray> by lazy {
        val bi = ImageIO.read(File((System.getenv("WM") ?: (System.getenv("HOME") + "/wm")) + "/data/real_in/r30_c253aa55.jpg"))
        val w = CW; val h = CH
        val ch = Array(3) { FloatArray(w * h) }
        for (y in 0 until h) for (x in 0 until w) { val p = bi.getRGB(X0 + x, Y0 + y); ch[0][y * w + x] = ((p shr 16) and 255).toFloat(); ch[1][y * w + x] = ((p shr 8) and 255).toFloat(); ch[2][y * w + x] = (p and 255).toFloat() }
        // 예전 앱이 찍은 빨간 점 지우기(점 주변 평균으로 덮음)
        for (y in 0 until h) for (x in 0 until w) { val r = ch[0][y * w + x]; val g = ch[1][y * w + x]; val b = ch[2][y * w + x]
            if (r > 200 && g < 40 && b < 40) { for (c in 0..2) { var sm = 0f; var n = 0; for (dy in -16..16 step 4) for (dx in -16..16 step 4) { val xx = x + dx; val yy = y + dy; if (xx in 0 until w && yy in 0 until h) { val rr = ch[0][yy * w + xx]; val gg = ch[1][yy * w + xx]; val bb = ch[2][yy * w + xx]; if (!(rr > 200 && gg < 40 && bb < 40)) { sm += ch[c][yy * w + xx]; n++ } } }; if (n > 0) ch[c][y * w + x] = sm / n } } }
        fun box(a: FloatArray, r: Int): FloatArray { val t = FloatArray(a.size); val o = FloatArray(a.size)
            for (y in 0 until h) { var sm = 0f; for (x in -r..r) sm += a[y * w + x.coerceIn(0, w - 1)]; for (x in 0 until w) { t[y * w + x] = sm / (2 * r + 1); sm += a[y * w + (x + r + 1).coerceIn(0, w - 1)] - a[y * w + (x - r).coerceIn(0, w - 1)] } }
            for (x in 0 until w) { var sm = 0f; for (y in -r..r) sm += t[y.coerceIn(0, h - 1) * w + x]; for (y in 0 until h) { o[y * w + x] = sm / (2 * r + 1); sm += t[(y + r + 1).coerceIn(0, h - 1) * w + x] - t[(y - r).coerceIn(0, h - 1) * w + x] } }
            return o }
        Array(3) { c -> var b = ch[c]; repeat(3) { b = box(b, 7) }; FloatArray(w * h) { (ch[c][it] - b[it]).coerceIn(-130f, 130f) } }
    }
    fun y0(k: Int, hh: Int, fixed: Boolean) = if (fixed) 700 else (80 + (k * 197 + 53) % (hh - CH - 160))
    fun apply(j: IntArray, hh: Int, k: Int, fixed: Boolean) {
        val y0 = y0(k, hh, fixed); val x0 = 840
        for (y in 0 until CH) for (x in 0 until CW) { val gx = x0 + x; val gy = y0 + y; if (gx >= W || gy >= hh) continue
            val p = j[gy * W + gx]; val i = y * CW + x
            val r = (((p shr 16) and 255) + hp[0][i]).toInt().coerceIn(0, 255); val g = (((p shr 8) and 255) + hp[1][i]).toInt().coerceIn(0, 255); val b = ((p and 255) + hp[2][i]).toInt().coerceIn(0, 255)
            j[gy * W + gx] = (255 shl 24) or (r shl 16) or (g shl 8) or b }
    }
}

fun main() {
    if (System.getProperty("wm.only") == "J") { runJ(); return }
    if (System.getProperty("wm.only") == "C") {
        val st = loadStamp("stamp_u.png", "stamp_L.png"); val st2 = loadStamp("stamp2_u.png", "stamp2_L.png")
        val f = java.io.File("models_c.bin"); f.delete()
        val seqA = listOf("white", "art0", "white", "art1", "navy", "art2", "black", "art3", "beige", "art4", "white", "art5", "art6", "white", "art0", "art2")
        val e = mkEngine(JvmCodec(), f)
        runSeq("A: 위 기준", e, seqA, st, 1490, 932, 720, show = false)
        e.persist(); Thread.sleep(800)
        val e2 = mkEngine(JvmCodec(), f); Thread.sleep(800)
        runSeq("B: 새 화", e2, listOf("art3", "art1", "white", "art4"), st, 1490, 932, 720, show = false)
        runSeq("C: 새 워터마크", e2, listOf("white", "art0", "art1", "white", "art2", "navy", "art3", "black", "art4", "white", "art5", "art6", "white", "art1"), st2, 1490, 700, 900)
        println(e2.fullDiag())
        return
    }
    if (System.getProperty("wm.only") == "G") {
        val st = loadStamp("stamp_u.png", "stamp_L.png")
        val seqA = listOf("white", "art0", "white", "art1", "navy", "art2", "black", "art3", "beige", "art4", "white", "art5", "art6", "white", "art0", "art2")
        val hs = listOf(1490, 1100, 1300, 900, 1490, 1200, 1000, 1400, 1250, 1100, 1490, 950, 1350, 1150, 1490, 1000)
        runSeq("G: 자유 y", mkEngine(JvmCodec(), null), seqA + listOf("art1", "white", "art3", "navy"), st, 0, 932, 0, heights = hs + listOf(1200, 1300, 1000, 1400), rnd = true)
        return
    }
    if (System.getProperty("wm.only") == "EF") {
        val st = loadStamp("stamp_u.png", "stamp_L.png")
        val seqA = listOf("white", "art0", "white", "art1", "navy", "art2", "black", "art3", "beige", "art4", "white", "art5", "art6", "white", "art0", "art2")
        val hs = listOf(1490, 1100, 1300, 900, 1490, 1200, 1000, 1400, 1250, 1100, 1490, 950, 1350, 1150, 1490, 1000)
        runSeq("E: BL", mkEngine(JvmCodec(), null), seqA, st, 0, 932, 0, bottomDist = 200, heights = hs)
        runSeq("F f=0.75", mkEngine(JvmCodec(), null), seqA, st, 0, 932, 0, heights = hs, frac = 0.75)
        return
    }
    if (System.getProperty("wm.only") == "AG" || System.getProperty("wm.only") == "AGF") {
        val fixed = System.getProperty("wm.only") == "AGF"
        val seq = (0 until 24).map { if (it % 5 == 1) "white" else "art${it % 7}" }
        val e = mkEngine(JvmCodec(), null)
        runSeq(if (fixed) "AGF: 사용자 붉은 도장(위치 고정)" else "AG: 사용자 붉은 도장(세로 위치 제각각)", e, seq, null, 0, 840, 0, heights = List(24) { 1100 + (it * 137) % 700 }, deco = { j, hh, k -> RedStamp.apply(j, hh, k, fixed) })
        println(e.fullDiag())
        return
    }
    if (System.getProperty("wm.only") == "FF") {
        val st = loadStamp("faint_u.png", "faint_L.png")
        val seq = (0 until 24).map { if (it % 5 == 1) "white" else "art${it % 7}" }
        val e = mkEngine(JvmCodec(), null)
        runSeq("FF: 프레임+워터마크(고정 위치)", e, seq, st, 1490, 900, 720, deco = ::frameDeco)
        println(e.fullDiag())
        return
    }
    if (System.getProperty("wm.only") == "FR" || System.getProperty("wm.only") == "FS") {
        val withStamp = System.getProperty("wm.only") == "FS"
        val st = loadStamp("faint_u.png", "faint_L.png")
        val seq = (0 until 24).map { if (it % 5 == 1) "white" else "art${it % 7}" }
        val e = mkEngine(JvmCodec(), null)
        runSeq(if (withStamp) "FS: 프레임+워터마크(고정 x)" else "FR: 프레임만(워터마크 없음)", e, seq, if (withStamp) st else null, 0, 900, 0, heights = List(24) { 1100 + (it * 137) % 700 }, rnd = true, deco = ::frameDeco)
        println(e.fullDiag())
        return
    }
    if (System.getProperty("wm.only") == "S") {
        val st = loadStamp("faint_u.png", "faint_L.png")
        val seq = (0 until 30).map { if (it % 5 == 1) "white" else "art${it % 7}" }
        val sxs = (0 until 30).map { if (System.getProperty("wm.fixedx") != null || it % 2 == 0) 965 else 85 }
        runSeq("S: 연한 실제 모양, 좌우 번갈아", mkEngine(JvmCodec(), null), seq, st, 0, 965, 0, heights = List(30) { 900 + (it * 137) % 700 }, rnd = true, sxs = sxs)
        return
    }
    if (System.getProperty("wm.only") == "R") {
        val st = loadStamp("stamp_u.png", "stamp_L.png")
        val faint = Stamp(st.w, st.h, FloatArray(st.u.size) { st.u[it] * (System.getProperty("wm.faint") ?: "0.45").toFloat() }, FloatArray(st.l.size) { 205f })
        val seqR = (0 until 30).map { if (it % 4 == 1) "white" else "art${it % 7}" }
        val sxsR = (0 until 30).map { if (System.getProperty("wm.fixedx") != null || it % 2 == 0) 932 else 36 }
        runSeq("R: 연한 워터마크, 좌우 번갈아·세로 제각각", mkEngine(JvmCodec(), null), seqR, faint, 0, 932, 0, heights = List(30) { 900 + (it * 137) % 700 }, rnd = true, sxs = sxsR)
        return
    }
    if (System.getProperty("wm.only") == "Q") {
        val st = loadStamp("stamp_u.png", "stamp_L.png")
        val red = Stamp(st.w, st.h, FloatArray(st.u.size) { Math.min(0.85f, st.u[it] * 0.9f) }, st.l, arrayOf(FloatArray(st.l.size) { 205f }, FloatArray(st.l.size) { 30f }, FloatArray(st.l.size) { 40f }))
        val seqQ = (0 until 36).map { "art${it % 7}" }
        val eq = mkEngine(JvmCodec(), null)
        runSeq("Q: 단색 배경 없이 그림 위에만 있는 붉은 워터마크(같은 자리)", eq, seqQ, red, 1490, 932, 720)
        println(eq.fullDiag())
        return
    }
    if (System.getProperty("wm.only") == "P") {
        val st = loadStamp("stamp_u.png", "stamp_L.png")
        val sc = (System.getProperty("wm.uscale") ?: "0.7").toFloat()
        fun f(c: Int) = FloatArray(st.w * st.h) { i -> val x = i % st.w; val y = i / st.w; (128 + 120 * when (c) { 0 -> Math.sin(0.045 * x + 0.02 * y); 1 -> Math.sin(0.035 * y + 1.0); else -> Math.cos(0.03 * (x + y)) }).toFloat() }
        val multi = Stamp(st.w, st.h, FloatArray(st.u.size) { Math.min(0.85f, st.u[it] * sc) }, st.l, arrayOf(f(0), f(1), f(2)))
        val seqP = (0 until 24).map { if (it % 5 == 1) "navy" else if (it % 5 == 3) "white" else if (it % 3 == 0) "art6" else "art${1 + it % 5}" }
        runSeq("P: 여러 색이 섞인 워터마크 (u x" + sc + ")", mkEngine(JvmCodec(), null), seqP, multi, 0, 932, 0, heights = List(24) { 900 + (it * 137) % 700 }, rnd = true)
        return
    }
    if (System.getProperty("wm.only") == "N") {
        val st = loadStamp("stamp_u.png", "stamp_L.png")
        val red = Stamp(st.w, st.h, FloatArray(st.u.size) { Math.min(0.85f, st.u[it] * 0.75f) }, st.l, arrayOf(FloatArray(st.l.size) { 205f }, FloatArray(st.l.size) { 30f }, FloatArray(st.l.size) { 40f }))
        val seqN = (0 until 24).map { if (it % 5 == 1) "navy" else if (it % 5 == 3) "black" else if (it % 2 == 0) "art6" else "art5" }
        runSeq("N: 약한 붉은 워터마크, 어두운 배경", mkEngine(JvmCodec(), null), seqN, red, 0, 932, 0, heights = List(24) { 900 + (it * 137) % 700 }, rnd = true)
        return
    }
    if (System.getProperty("wm.only") == "M") {
        val st = loadStamp("stamp_u.png", "stamp_L.png")
        val red = Stamp(st.w, st.h, FloatArray(st.u.size) { Math.min(0.85f, st.u[it] * 1.15f) }, st.l, arrayOf(FloatArray(st.l.size) { 205f }, FloatArray(st.l.size) { 30f }, FloatArray(st.l.size) { 40f }))
        val seqM = (0 until 24).map { if (it % 4 == 1) "white" else if (it % 3 == 0) "art6" else "art${1 + it % 5}" }
        runSeq("M: 붉은 워터마크", mkEngine(JvmCodec(), null), seqM, red, 0, 932, 0, heights = List(24) { 900 + (it * 137) % 700 }, rnd = true)
        return
    }
    if (System.getProperty("wm.only") == "L") {
        val st = loadStamp("stamp_u.png", "stamp_L.png")
        val sc = 1.15; val nw = (st.w * sc).toInt(); val nh = (st.h * sc).toInt()
        val stv = Stamp(nw, nh, FloatArray(nw * nh) { i -> Math.min(0.95f, st.u[Math.min(st.h - 1, (i / nw / sc).toInt()) * st.w + Math.min(st.w - 1, (i % nw / sc).toInt())] * 1.6f) },
            FloatArray(nw * nh) { i -> st.l[Math.min(st.h - 1, (i / nw / sc).toInt()) * st.w + Math.min(st.w - 1, (i % nw / sc).toInt())] * 0.7f })
        val e = mkEngine(JvmCodec(), null)
        val seqA = listOf("white", "art0", "white", "art1", "navy", "art2", "black", "art3", "beige", "art4", "white", "art5", "art6", "white", "art0", "art2")
        runSeq("L1: 원래 워터마크", e, seqA, st, 1490, 932, 720)
        e.onPage()
        runSeq("L2: 같은 자리, 더 진하고 15% 큰 변형", e, seqA, stv, 1490, 932, 720)
        println(e.fullDiag())
        return
    }
    if (System.getProperty("wm.only") == "K") {
        val st = loadStamp("stamp_u.png", "stamp_L.png")
        val e = mkEngine(JvmCodec(), null)
        val seqJ = (0 until 14).map { if (it % 4 == 1) "white" else if (it % 2 == 0) "art6" else "art5" }
        runSeq("K1: 워터마크 있음", e, seqJ, st, 0, 932, 0, heights = List(14) { 900 + (it * 137) % 700 }, rnd = true)
        val seqN = (0 until 20).map { if (it % 4 == 1) "white" else "art${it % 7}" }
        runSeq("K2: 같은 너비 워터마크 없는 이미지", e, seqN, null, 0, 932, 0, heights = List(20) { 900 + (it * 137) % 700 }, rnd = true)
        return
    }
    if (System.getProperty("wm.only") == "HI") {
        val st = loadStamp("stamp_u.png", "stamp_L.png")
        val seqH = (0 until 30).map { if (it % 4 == 1) "white" else "art${1 + (it * 3) % 4}" }
        runSeq("H: 흰+그림", mkEngine(JvmCodec(), null), seqH, st, 0, 932, 0, heights = List(30) { 900 + (it * 137) % 700 }, rnd = true)
        val stc = Stamp(st.w, st.h, FloatArray(st.u.size) { Math.min(0.9f, st.u[it] * (255f - st.l[it]) / 155f) }, FloatArray(st.l.size) { 100f })
        runSeq("I: 단일 L", mkEngine(JvmCodec(), null), seqH, stc, 0, 932, 0, heights = List(30) { 900 + (it * 137) % 700 }, rnd = true)
        return
    }
    val st = loadStamp("stamp_u.png", "stamp_L.png"); val st2 = loadStamp("stamp2_u.png", "stamp2_L.png")
    val f = File("models.bin"); f.delete()
    val e = mkEngine(JvmCodec(), f)
    val seqA = listOf("white", "art0", "white", "art1", "navy", "art2", "black", "art3", "beige", "art4", "white", "art5", "art6", "white", "art0", "art2")
    runSeq("A: 위 기준, 같은 높이, 배경 다양", e, seqA, st, 1490, 932, 720)
    e.persist(); Thread.sleep(800)
    println("-- B: 저장된 모델로 새 엔진(앱 재시작) --")
    val e2 = mkEngine(JvmCodec(), f); Thread.sleep(800)
    runSeq("B: 새 화(첫 장부터)", e2, listOf("art3", "art1", "white", "art4"), st, 1490, 932, 720)
    println("-- C: 워터마크가 바뀜(다른 모양·위치) --")
    runSeq("C: 새 워터마크", e2, listOf("white", "art0", "art1", "white", "art2", "navy", "art3", "black", "art4", "white", "art5", "art6", "white", "art1"), st2, 1490, 700, 900)
    println("-- D: 워터마크 없음 --")
    val e3 = mkEngine(JvmCodec(), null)
    runSeq("D: 없음", e3, listOf("white", "art0", "art1", "white", "art2", "navy", "art3", "black", "art4", "white", "art5", "art6", "white", "art1"), null, 1490, 932, 720)
    println("-- E: 아래 기준(높이가 제각각) --")
    val e4 = mkEngine(JvmCodec(), null)
    val hs = listOf(1490, 1100, 1300, 900, 1490, 1200, 1000, 1400, 1250, 1100, 1490, 950, 1350, 1150, 1490, 1000)
    runSeq("E: BL", e4, seqA, st, 0, 932, 0, bottomDist = 200, heights = hs)
    println("-- G: x 같고 y 제각각 --")
    val e6 = mkEngine(JvmCodec(), null)
    runSeq("G: 자유 y", e6, seqA + listOf("art1", "white", "art3", "navy"), st, 0, 932, 0, heights = hs + listOf(1200, 1300, 1000, 1400), rnd = true)
    println("-- H: 흰 배경 + 그림만(색 다양성 없음), y 제각각 --")
    val e7 = mkEngine(JvmCodec(), null)
    val seqH = (0 until 30).map { if (it % 4 == 1) "white" else "art${1 + (it * 3) % 4}" }
    runSeq("H: 흰+그림", e7, seqH, st, 0, 932, 0, heights = List(30) { 900 + (it * 137) % 700 }, rnd = true)
    println(e7.fullDiag())
    println("-- I: 로고 색이 하나(L=100)인 워터마크, 흰+그림만 --")
    val stc = Stamp(st.w, st.h, FloatArray(st.u.size) { Math.min(0.9f, st.u[it] * (255f - st.l[it]) / 155f) }, FloatArray(st.l.size) { 100f })
    val e8 = mkEngine(JvmCodec(), null)
    runSeq("I: 단일 L", e8, seqH, stc, 0, 932, 0, heights = List(30) { 900 + (it * 137) % 700 }, rnd = true)
    println(e8.fullDiag())
    for (f in listOf(0.5, 0.25, 0.75)) {
        println("-- F: 높이에 비례한 위치 f=$f --")
        val e5 = mkEngine(JvmCodec(), null)
        runSeq("F f=$f", e5, seqA, st, 0, 932, 0, heights = hs, frac = f)
    }
}

fun runJ() {
    val st = loadStamp("stamp_u.png", "stamp_L.png")
    println("-- J: 어두운 그림 섞임, 실제 워터마크 --")
    val e9 = mkEngine(JvmCodec(), null)
    val seqJ = (0 until 30).map { if (it % 4 == 1) "white" else if (it % 2 == 0) "art6" else "art5" }
    runSeq("J: 어두운 그림", e9, seqJ, st, 0, 932, 0, heights = List(30) { 900 + (it * 137) % 700 }, rnd = true)
    println(e9.fullDiag())
}

class JvmNet(path: String) : WmNet {
    private val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
    private val session = env.createSession(java.io.File(path).readBytes(), ai.onnxruntime.OrtSession.SessionOptions())
    override fun infer(x: FloatArray, h: Int, w: Int): FloatArray? {
        ai.onnxruntime.OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(x), longArrayOf(1, 9, h.toLong(), w.toLong())).use { t ->
            session.run(mapOf("x" to t)).use { r ->
                val fb = (r[0] as ai.onnxruntime.OnnxTensor).floatBuffer
                val out = FloatArray(fb.remaining()); fb.get(out); return out
            }
        }
    }
}
val GNN: WmNet? = System.getProperty("wm.nn")?.let { JvmNet(it) }
fun mkEngine(c: WmCodec, f: java.io.File?): WmEngine = WmEngine(c, f).also { it.nn = GNN; if (System.getProperty("wm.cleanoff") != null) it.enabled = false }
