package com.pixelpeeper

import java.nio.ByteBuffer

/**
 * Android JNI facade for the native PixelPeeper engine.
 *
 * Put libpixelpeeper.so and libpixelpeeper_jni.so in jniLibs/<abi>/.
 * The input must be a direct RGB888 buffer (for example a Bitmap copied with
 * copyPixelsToBuffer into a direct ByteBuffer).
 */
object NativeAnalyzer {
    init {
        System.loadLibrary("pixelpeeper")
        System.loadLibrary("pixelpeeper_jni")
    }

    @JvmStatic
    private external fun nativeAnalyze(
        rgb: ByteBuffer,
        width: Int,
        height: Int,
        stride: Int,
        roi: IntArray?,
        rgba: Boolean,
        quick: Boolean,
        out: DoubleArray
    ): Int

    data class Evidence(
        val method: String,
        val scale: Double,
        val period: Int,
        val support: Int,
        val prominence: Double
    )

    data class Candidate(
        val width: Int,
        val height: Int,
        val scale: Double,
        val score: Double,
        val exact: Boolean
    ) {
        val label: String get() = "${width}x$height"
    }

    data class Result(
        val width: Int,
        val height: Int,
        val scaleX: Double,
        val scaleY: Double,
        val scale: Double,
        val internalWidth: Int,
        val internalHeight: Int,
        val percent: Double,
        val method: String,
        val confidence: Double,
        val nativeLikely: Boolean,
        val bars: IntArray,
        val evidenceX: Evidence,
        val evidenceY: Evidence,
        val candidates: List<Candidate>,
        val elapsedMs: Double
    )

    /**
     * @param rgb direct ByteBuffer positioned at the first pixel byte
     * @param stride row stride in bytes, usually width * 3 (RGB) or width * 4 (RGBA)
     * @param roi optional intArrayOf(x0, y0, x1, y1) in full-image pixels
     * @param rgba set true for RGBA8888, false for RGB888
     */
    fun analyze(
        rgb: ByteBuffer,
        width: Int,
        height: Int,
        roi: IntArray? = null,
        rgba: Boolean = false,
        stride: Int = width * if (rgba) 4 else 3,
        quick: Boolean = false
    ): Result {
        require(rgb.isDirect) { "rgb must be a direct ByteBuffer" }
        val out = DoubleArray(80)
        val rc = nativeAnalyze(rgb, width, height, stride, roi, rgba, quick, out)
        check(rc == 0) { "pixelpeeper analysis failed (rc=$rc)" }

        fun methodName(v: Double): String = when (v.toInt()) {
            1 -> "nearest"
            2 -> "smooth"
            else -> "none"
        }
        fun evidence(base: Int) = Evidence(
            method = methodName(out[base]),
            scale = out[base + 1],
            period = out[base + 2].toInt(),
            support = out[base + 5].toInt(),
            prominence = out[base + 8]
        )

        val count = out[37].toInt().coerceIn(0, 8)
        val candidates = ArrayList<Candidate>(count)
        for (i in 0 until count) {
            val p = 38 + i * 5
            candidates += Candidate(
                width = out[p].toInt(),
                height = out[p + 1].toInt(),
                scale = out[p + 2],
                score = out[p + 3],
                exact = out[p + 4].toInt() == 1
            )
        }
        return Result(
            width = out[0].toInt(),
            height = out[1].toInt(),
            scaleX = out[2],
            scaleY = out[3],
            scale = out[4],
            internalWidth = out[5].toInt(),
            internalHeight = out[6].toInt(),
            percent = out[7],
            method = methodName(out[8]),
            confidence = out[9],
            nativeLikely = out[10].toInt() != 0,
            bars = intArrayOf(out[11].toInt(), out[12].toInt(), out[13].toInt(), out[14].toInt()),
            evidenceX = evidence(15),
            evidenceY = evidence(26),
            candidates = candidates,
            elapsedMs = out[78]
        )
    }
}
