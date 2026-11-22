package com.example.comicviewer

/** 디코드된 이미지(읽기 전용 픽셀 배열) */
class WmImg(val w: Int, val h: Int, val px: IntArray)

/** 디코드된 전체 이미지(영역을 읽고 쓸 수 있음) */
interface WmFull {
    val w: Int
    val h: Int
    fun getRegion(x: Int, y: Int, rw: Int, rh: Int, out: IntArray)
    fun setRegion(x: Int, y: Int, rw: Int, rh: Int, px: IntArray)
    fun encode(png: Boolean): ByteArray
    fun close()
}

/** 이미지 디코드/인코드(안드로이드는 BitmapFactory, PC 시험은 ImageIO) */
interface WmCodec {
    fun info(bytes: ByteArray): IntArray?            // [너비, 높이]
    fun small(bytes: ByteArray, s: Int): WmImg?      // 1/s 로 줄여 디코드
    fun full(bytes: ByteArray): WmFull?
}
