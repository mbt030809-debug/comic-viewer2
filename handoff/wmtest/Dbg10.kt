import com.example.comicviewer.*
import java.io.File
fun main(args: Array<String>) {
    val c = JvmCodec()
    val names = args.toList()
    val ts = names.map { n -> val b = File("/tmp/real_crop/$n.jpg").readBytes(); val d = c.info(b)!!; val s = WmCore.scaleFor(d[0]); val sm = c.small(b, s)!!; WmCore.makeThumb(sm, s, d[0], d[1]) }
    println("calm: " + ts.map { it.calm })
    val det = WmCore.detect(ts); println("고정형: " + (det?.let { "frame=${it.frame} tx0=${it.tx0 * ts[0].s} y=${it.tyF * ts[0].s} box=${it.tbw * ts[0].s}x${it.tbh * ts[0].s} score=${it.score}" } ?: "없음") + " lastMax=${WmCore.lastMax}")
    val df = WmCore.detectFree(ts); println("가변형: " + (df?.let { "tx0=${it.tx0 * ts[0].s} box=${it.tbw * ts[0].s}x${it.tbh * ts[0].s} locs=${it.locs?.map { l -> l * ts[0].s }} score=${it.score}" } ?: "없음") + " dbg=${WmCore.dbgFree}")
}
