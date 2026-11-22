package com.example.comicviewer

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import android.webkit.CookieManager
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.FutureTask
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlin.math.max

/** 안드로이드용 이미지 디코드/인코드 */
class AndroidCodec : WmCodec {
    override fun info(bytes: ByteArray): IntArray? {
        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, o)
        return if (o.outWidth > 0 && o.outHeight > 0) intArrayOf(o.outWidth, o.outHeight) else null
    }

    override fun small(bytes: ByteArray, s: Int): WmImg? {
        val o = BitmapFactory.Options().apply { inSampleSize = s; inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val b = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, o) ?: return null
        val px = IntArray(b.width * b.height)
        b.getPixels(px, 0, b.width, 0, 0, b.width, b.height)
        val r = WmImg(b.width, b.height, px); b.recycle()
        return r
    }

    override fun full(bytes: ByteArray): WmFull? {
        val o = BitmapFactory.Options().apply { inMutable = true; inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val b = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, o) ?: return null
        return object : WmFull {
            override val w get() = b.width
            override val h get() = b.height
            override fun getRegion(x: Int, y: Int, rw: Int, rh: Int, out: IntArray) { b.getPixels(out, 0, rw, x, y, rw, rh) }
            override fun setRegion(x: Int, y: Int, rw: Int, rh: Int, px: IntArray) { b.setPixels(px, 0, rw, x, y, rw, rh) }
            override fun encode(png: Boolean): ByteArray {
                val bo = ByteArrayOutputStream(max(1 shl 16, b.byteCount / 6))
                b.compress(if (png) Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG, if (png) 100 else 95, bo)
                return bo.toByteArray()
            }
            override fun close() { b.recycle() }
        }
    }
}

/**
 * 웹뷰의 이미지 요청을 가로채 내려받고, 워터마크를 지운 이미지를 대신 돌려줌.
 * 어떤 단계든 실패하면 null(= 웹뷰가 원래대로 직접 받음). 같은 주소는 메모리 캐시/중복 요청 합치기.
 */
class WmGlue(val engine: WmEngine) {
    @Volatile var pageUrl = ""
    @Volatile var reqs = 0    // 가로챈 요청 수(진단용)
    @Volatile var imgs = 0    // 그중 이미지로 보이는 요청 수
    @Volatile var ok = 0      // 내려받아 처리한 이미지 수
    @Volatile var bad = 0     // 내려받기 실패 수
    @Volatile var lastFail = ""   // 마지막 실패 원인
    private class Out(val bytes: ByteArray, val mime: String, val cleaned: Boolean = false)
    private val plain = ConcurrentHashMap.newKeySet<String>()   // 워터마크를 못 지우고 원본 그대로 내보낸 이미지 주소(나중에 모양을 찾으면 다시 받아 지움)
    private val cache = object : LruCache<String, Out>(40 * 1024 * 1024) { override fun sizeOf(key: String, value: Out) = value.bytes.size }
    private val inflight = ConcurrentHashMap<String, FutureTask<Out?>>()
    private val netSem = Semaphore(8)                                // 동시 내려받기 제한(연결 보호)
    private val cpuSem = Semaphore(3)                                // 동시 가공(디코드·제거·인코드) 제한(메모리·발열 보호)
    private val fails = ConcurrentHashMap<String, LongArray>()      // 호스트 → [연속 실패 횟수, 마지막 실패 시각]

    fun intercept(r: WebResourceRequest): WebResourceResponse? {
        if (!engine.wants()) return null
        if (r.isForMainFrame || r.method != "GET") return null
        reqs++
        val sc = r.url.scheme; if (sc != "http" && sc != "https") return null
        val h = r.requestHeaders ?: emptyMap()
        if (h.keys.any { it.equals("Range", true) }) return null     // 부분 요청은 웹뷰가 처리
        val path = r.url.path?.lowercase() ?: ""
        if (path.endsWith(".svg") || path.endsWith(".gif") || path.endsWith(".ico")) return null
        val accept = h.entries.firstOrNull { it.key.equals("Accept", true) }?.value ?: ""
        val ext = path.endsWith(".jpg") || path.endsWith(".jpeg") || path.endsWith(".png") || path.endsWith(".webp") || path.endsWith(".jfif") || path.endsWith(".bmp") || path.endsWith(".avif")
        val noExt = !path.substringAfterLast('/').contains('.')
        if (!accept.contains("image/") && !ext && !(accept.isEmpty() && noExt)) return null // Accept 헤더가 없고 확장자도 없으면(주소만으로 판단 불가) 일단 받아서 확인
        imgs++
        val host = r.url.host ?: return null
        fails[host]?.let { if (it[0] >= 3 && System.currentTimeMillis() - it[1] < 60_000) return null }
        val url = stripRetro(r.url.toString())
        cache.get(url)?.let { return respond(it) }
        val task = FutureTask<Out?> { fetch(url, h, host) }
        val prev = inflight.putIfAbsent(url, task)
        val t = prev ?: task.also { it.run() }
        val out = try { t.get(75, TimeUnit.SECONDS) } catch (_: Throwable) { null }
        if (prev == null) inflight.remove(url)
        return out?.let { respond(it) }
    }

    fun clearCache() { cache.evictAll(); plain.clear() }

    /** 새 페이지: 지우지 못한 채 캐시된 이미지는 버림(다시 열면 이번엔 지워서 내보내도록) */
    fun newPage() {
        plain.clear()
        for ((k, v) in cache.snapshot()) if (!v.cleaned) cache.remove(k)
    }

    /** 워터마크 모양을 찾은 뒤 호출: 그동안 원본 그대로 내보낸 이미지들의 캐시를 비우고 그 주소 목록을 돌려줌(페이지가 다시 받으면 이번엔 지워서 내보냄) */
    fun takePlain(): List<String> {
        val l = plain.toList(); plain.clear()
        for (u in l) { val c = cache.get(u); if (c != null && !c.cleaned) cache.remove(u) }
        return l
    }

    private fun respond(o: Out): WebResourceResponse {
        val hdr = HashMap<String, String>(); hdr["Access-Control-Allow-Origin"] = "*"; hdr["Cache-Control"] = "no-store"   // 가공한 이미지(점·상자가 구워짐)를 웹뷰가 캐시하면 스위치를 꺼도 reload 때 그대로 보이므로 보관하지 않게 함(앱 안 캐시가 빠르게 응답)
        return WebResourceResponse(o.mime, null, 200, "OK", hdr, ByteArrayInputStream(o.bytes))
    }

    private fun fetch(url: String, reqHeaders: Map<String, String>, host: String): Out? {
        try {
            val d: Pair<ByteArray, String?>
            try { netSem.acquire() } catch (_: InterruptedException) { return null }
            try { d = download(url, reqHeaders) } finally { netSem.release() }
            val bytes = d.first; val mime = mimeOf(d.second, bytes)
            if (!mime.startsWith("image/")) { lastFail = "이미지 아님 " + (d.second ?: "?") + " " + host; return null }
            fails.remove(host)
            var processed: ByteArray? = null
            if (mime == "image/jpeg" || mime == "image/png" || mime == "image/webp") {
                try { cpuSem.acquire() } catch (_: InterruptedException) { return null }
                try { processed = engine.process(bytes, mime) } finally { cpuSem.release() }
            }
            val out = if (processed != null) Out(processed, if (mime == "image/png") "image/png" else "image/jpeg", true) else Out(bytes, mime)
            cache.put(url, out); ok++
            if (processed == null && (mime == "image/jpeg" || mime == "image/png" || mime == "image/webp") && plain.size < 300) plain.add(url)
            return out
        } catch (t: Throwable) { fail(host); bad++; lastFail = (t.message ?: t.javaClass.simpleName).take(60) + " " + host; return null }
    }

    private fun fail(host: String) { val f = fails.getOrPut(host) { longArrayOf(0, 0) }; f[0]++; f[1] = System.currentTimeMillis() }

    private fun mimeOf(ct: String?, b: ByteArray): String {
        val c = ct?.substringBefore(';')?.trim()?.lowercase() ?: ""
        if (c.startsWith("image/")) return c
        if (b.size > 12) {
            if (b[0] == 0xFF.toByte() && b[1] == 0xD8.toByte()) return "image/jpeg"
            if (b[0] == 0x89.toByte() && b[1] == 'P'.code.toByte()) return "image/png"
            if (b[0] == 'R'.code.toByte() && b[8] == 'W'.code.toByte() && b[9] == 'E'.code.toByte()) return "image/webp"
        }
        return c
    }

    /** 요청 헤더(UA·Referer 등)를 그대로 쓰고 쿠키는 웹뷰와 공유. http→https 리다이렉트는 직접 따라감 */
    private fun download(url0: String, reqHeaders: Map<String, String>): Pair<ByteArray, String?> {
        var url = url0; var hops = 0
        while (hops++ <= 5) {
            val conn = URL(url).openConnection() as HttpURLConnection
            conn.instanceFollowRedirects = false
            conn.connectTimeout = 10000; conn.readTimeout = 20000
            var hasUa = false; var hasRef = false
            for ((k, v) in reqHeaders) {
                val kl = k.lowercase()
                if (kl == "cookie" || kl == "host" || kl == "connection" || kl == "accept-encoding" || kl == "range" || kl == "if-none-match" || kl == "if-modified-since" || kl == "content-length") continue
                if (kl == "user-agent") hasUa = true
                if (kl == "referer") hasRef = true
                conn.setRequestProperty(k, v)
            }
            if (!hasUa) conn.setRequestProperty("User-Agent", BandAnalyzer.userAgent)
            if (!hasRef && pageUrl.isNotEmpty()) conn.setRequestProperty("Referer", pageUrl)
            if (!reqHeaders.keys.any { it.equals("Accept", true) }) conn.setRequestProperty("Accept", "image/avif,image/webp,image/apng,image/*,*/*;q=0.8")
            CookieManager.getInstance().getCookie(url)?.let { conn.setRequestProperty("Cookie", it) }
            val code = conn.responseCode
            conn.headerFields?.get("Set-Cookie")?.forEach { CookieManager.getInstance().setCookie(url, it) }
            if (code in 300..399) {
                val loc = conn.getHeaderField("Location") ?: throw java.io.IOException("리다이렉트 주소 없음")
                url = URL(URL(url), loc).toString(); conn.disconnect(); continue
            }
            if (code != 200) { conn.disconnect(); throw java.io.IOException("HTTP $code") }
            val ct = conn.contentType
            val cl = conn.contentLength
            val out = ByteArrayOutputStream(if (cl in 1..25_000_000) cl else 1 shl 16)
            val buf = ByteArray(64 * 1024)
            conn.inputStream.use { ins ->
                while (true) {
                    val n = ins.read(buf); if (n < 0) break
                    out.write(buf, 0, n)
                    if (out.size() > 25_000_000) { conn.disconnect(); throw java.io.IOException("25MB 초과") }
                }
            }
            conn.disconnect()
            return out.toByteArray() to ct
        }
        throw java.io.IOException("리다이렉트 과다")
    }

    companion object {
        /** 다시 받기용으로 페이지가 덧붙인 _wmr 값을 떼어 원래 주소로 되돌림(사이트에는 덧붙인 값이 나가지 않음) */
        fun stripRetro(u: String): String {
            val i = u.indexOf("_wmr=")
            if (i < 1 || (u[i - 1] != '?' && u[i - 1] != '&')) return u
            var j = u.indexOf('&', i); if (j < 0) j = u.length else j += 1
            val hash = u.indexOf('#', i); if (hash in i until j) j = hash
            val head = u.substring(0, i); val tail = u.substring(j)
            val r = (head + tail).removeSuffix("?").removeSuffix("&")
            return r
        }
        fun create(dir: File): WmGlue = WmGlue(WmEngine(AndroidCodec(), File(dir, "wm_models.bin")))
    }
}
