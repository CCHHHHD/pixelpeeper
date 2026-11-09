/*
 * Optional Android JNI bridge.
 *
 * Java/Kotlin side (package com.pixelpeeper):
 *
 *   object NativeAnalyzer {
 *       init {
 *           System.loadLibrary("pixelpeeper")
 *           System.loadLibrary("pixelpeeper_jni")
 *       }
 *       external fun nativeAnalyze(
 *           rgb: java.nio.ByteBuffer, width: Int, height: Int, stride: Int,
 *           roi: IntArray?, quick: Boolean, out: DoubleArray): Int
 *   }
 *
 * The direct ByteBuffer must start at the first RGB byte. `out` must hold at
 * least 80 doubles; see NATIVE_OUT_* constants in the Kotlin sample.
 */
#include "pixelpeeper.h"

#include <jni.h>
#include <cstring>

namespace {
constexpr int kOutSize = 80;
constexpr int kEvidenceFields = 11;

void put_evidence(jdouble *out, int base, const pp_evidence &e) {
    out[base + 0] = static_cast<double>(e.method);
    out[base + 1] = e.scale;
    out[base + 2] = static_cast<double>(e.period);
    out[base + 3] = e.duty;
    out[base + 4] = e.match;
    out[base + 5] = static_cast<double>(e.support);
    out[base + 6] = e.spread;
    out[base + 7] = static_cast<double>(e.tiles);
    out[base + 8] = e.prominence;
    out[base + 9] = e.phase;
    out[base + 10] = static_cast<double>(e.near_native);
}
} // namespace

extern "C" JNIEXPORT jint JNICALL
Java_com_pixelpeeper_NativeAnalyzer_nativeAnalyze(JNIEnv *env, jclass, jobject rgb_buffer,
                                                  jint width, jint height, jint stride,
                                                  jintArray roi_array, jboolean rgba,
                                                  jboolean quick,
                                                  jdoubleArray out_array) {
    if (!rgb_buffer || !out_array) return -1;
    if (env->GetArrayLength(out_array) < kOutSize) return -1;
    if (env->GetDirectBufferCapacity(rgb_buffer) < 0) return -1;
    auto *base = static_cast<unsigned char *>(env->GetDirectBufferAddress(rgb_buffer));
    if (!base) return -1;
    const int channels = rgba == JNI_TRUE ? 4 : 3;
    if (stride < width * channels || width < 2 || height < 2) return -1;

    pp_roi roi{};
    pp_roi *roi_ptr = nullptr;
    if (roi_array && env->GetArrayLength(roi_array) >= 4) {
        jint values[4] = {0, 0, 0, 0};
        env->GetIntArrayRegion(roi_array, 0, 4, values);
        roi = {values[0], values[1], values[2], values[3]};
        roi_ptr = &roi;
    }

    pp_result result;
    const int rc = rgba == JNI_TRUE
                       ? pp_analyze_rgba(base, width, height, stride, roi_ptr,
                                         quick == JNI_TRUE ? 1 : 0, &result)
                       : pp_analyze_rgb(base, width, height, stride, roi_ptr,
                                        quick == JNI_TRUE ? 1 : 0, &result);
    jdouble values[kOutSize] = {0};
    values[0] = result.width;
    values[1] = result.height;
    values[2] = result.scale_x;
    values[3] = result.scale_y;
    values[4] = result.scale;
    values[5] = result.internal_w;
    values[6] = result.internal_h;
    values[7] = result.percent;
    values[8] = static_cast<double>(result.method);
    values[9] = result.confidence;
    values[10] = static_cast<double>(result.native_likely);
    for (int i = 0; i < 4; ++i) values[11 + i] = result.bars[i];
    put_evidence(values, 15, result.evidence_x);
    put_evidence(values, 26, result.evidence_y);
    values[37] = static_cast<double>(result.candidate_count);
    int at = 38;
    for (int i = 0; i < result.candidate_count && i < 8; ++i) {
        const auto &c = result.candidates[i];
        values[at++] = c.width;
        values[at++] = c.height;
        values[at++] = c.scale;
        values[at++] = c.score;
        values[at++] = static_cast<double>(c.kind);
    }
    values[78] = result.elapsed_ms;
    values[79] = static_cast<double>(rc);
    env->SetDoubleArrayRegion(out_array, 0, kOutSize, values);
    return rc;
}
