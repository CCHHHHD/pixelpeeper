/*
 * Optional HarmonyOS/OpenHarmony Node-API bridge.
 *
 * ArkTS side:
 *   import pixelpeeper from 'libpixelpeeper.so';
 *   const r = pixelpeeper.analyze(rgbArrayBuffer, width, height, stride, roi, quick, rgba);
 *
 * `rgba` defaults to false (RGB888). Set it to true for PixelMap RGBA_8888
 * buffers. Build this file together with the core into a shared library named
 * libpixelpeeper.so using the HarmonyOS native CMake toolchain.
 */
#include "pixelpeeper.h"

#include "napi/native_api.h"

#include <cstring>

namespace {

napi_value number(napi_env env, double v) {
    napi_value out = nullptr;
    napi_create_double(env, v, &out);
    return out;
}

napi_value boolean(napi_env env, bool v) {
    napi_value out = nullptr;
    napi_get_boolean(env, v, &out);
    return out;
}

napi_value str(napi_env env, const char *v) {
    napi_value out = nullptr;
    napi_create_string_utf8(env, v, NAPI_AUTO_LENGTH, &out);
    return out;
}

void set(napi_env env, napi_value obj, const char *key, napi_value value) {
    napi_set_named_property(env, obj, key, value);
}

napi_value evidence_object(napi_env env, const pp_evidence &e) {
    napi_value o = nullptr;
    napi_create_object(env, &o);
    const char *method = e.method == PP_METHOD_NEAREST ? "nearest"
                         : e.method == PP_METHOD_SMOOTH ? "smooth" : "none";
    set(env, o, "method", str(env, method));
    set(env, o, "scale", number(env, e.scale));
    set(env, o, "period", number(env, e.period));
    set(env, o, "duty", number(env, e.duty));
    set(env, o, "match", number(env, e.match));
    set(env, o, "support", number(env, e.support));
    set(env, o, "spread", number(env, e.spread));
    set(env, o, "tiles", number(env, e.tiles));
    set(env, o, "prominence", number(env, e.prominence));
    set(env, o, "phase", number(env, e.phase));
    set(env, o, "nearNative", boolean(env, e.near_native != 0));
    return o;
}

napi_value analyze(napi_env env, napi_callback_info info) {
    size_t argc = 7;
    napi_value argv[7] = {nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    if (argc < 3) {
        napi_throw_type_error(env, nullptr,
                              "analyze(buffer, width, height[, stride[, roi[, quick[, rgba]]]])");
        return nullptr;
    }

    void *data = nullptr;
    size_t length = 0;
    bool is_arraybuffer = false;
    napi_is_arraybuffer(env, argv[0], &is_arraybuffer);
    if (!is_arraybuffer || napi_get_arraybuffer_info(env, argv[0], &data, &length) != napi_ok) {
        napi_throw_type_error(env, nullptr, "first argument must be an RGB888/RGBA8888 ArrayBuffer");
        return nullptr;
    }
    int32_t width = 0, height = 0, stride = 0;
    napi_get_value_int32(env, argv[1], &width);
    napi_get_value_int32(env, argv[2], &height);
    bool rgba = false;
    if (argc >= 7) {
        napi_value b = nullptr;
        napi_coerce_to_bool(env, argv[6], &b);
        napi_get_value_bool(env, b, &rgba);
    }
    stride = width * (rgba ? 4 : 3);
    if (argc >= 4) {
        napi_valuetype t = napi_undefined;
        napi_typeof(env, argv[3], &t);
        if (t == napi_number) napi_get_value_int32(env, argv[3], &stride);
    }
    if (length < static_cast<size_t>(stride) * static_cast<size_t>(height)) {
        napi_throw_range_error(env, nullptr, "ArrayBuffer is smaller than stride * height");
        return nullptr;
    }

    pp_roi roi_value{};
    pp_roi *roi = nullptr;
    if (argc >= 5) {
        bool is_array = false;
        napi_is_array(env, argv[4], &is_array);
        if (is_array) {
            uint32_t n = 0;
            napi_get_array_length(env, argv[4], &n);
            if (n >= 4) {
                int32_t v[4] = {0, 0, 0, 0};
                for (uint32_t i = 0; i < 4; ++i) {
                    napi_value item = nullptr;
                    napi_get_element(env, argv[4], i, &item);
                    napi_get_value_int32(env, item, &v[i]);
                }
                roi_value = {v[0], v[1], v[2], v[3]};
                roi = &roi_value;
            }
        }
    }
    bool quick = false;
    if (argc >= 6) {
        napi_value b = nullptr;
        napi_coerce_to_bool(env, argv[5], &b);
        napi_get_value_bool(env, b, &quick);
    }

    pp_result r;
    const int rc = rgba
                       ? pp_analyze_rgba(static_cast<const uint8_t *>(data), width, height,
                                         stride, roi, quick ? 1 : 0, &r)
                       : pp_analyze_rgb(static_cast<const uint8_t *>(data), width, height,
                                        stride, roi, quick ? 1 : 0, &r);
    if (rc != 0) {
        napi_throw_error(env, nullptr, r.error[0] ? r.error : "pixelpeeper analysis failed");
        return nullptr;
    }

    napi_value out = nullptr;
    napi_create_object(env, &out);
    set(env, out, "width", number(env, r.width));
    set(env, out, "height", number(env, r.height));
    set(env, out, "scaleX", number(env, r.scale_x));
    set(env, out, "scaleY", number(env, r.scale_y));
    set(env, out, "scale", number(env, r.scale));
    set(env, out, "internalWidth", number(env, r.internal_w));
    set(env, out, "internalHeight", number(env, r.internal_h));
    set(env, out, "percent", number(env, r.percent));
    const char *method = r.method == PP_METHOD_NEAREST ? "nearest"
                         : r.method == PP_METHOD_SMOOTH ? "smooth" : "none";
    set(env, out, "method", str(env, method));
    set(env, out, "confidence", number(env, r.confidence));
    set(env, out, "nativeLikely", boolean(env, r.native_likely != 0));
    set(env, out, "elapsedMs", number(env, r.elapsed_ms));

    napi_value bars = nullptr;
    napi_create_array_with_length(env, 4, &bars);
    for (uint32_t i = 0; i < 4; ++i) {
        napi_set_element(env, bars, i, number(env, r.bars[i]));
    }
    set(env, out, "bars", bars);
    set(env, out, "evidenceX", evidence_object(env, r.evidence_x));
    set(env, out, "evidenceY", evidence_object(env, r.evidence_y));

    napi_value candidates = nullptr;
    napi_create_array_with_length(env, static_cast<size_t>(r.candidate_count), &candidates);
    for (int i = 0; i < r.candidate_count; ++i) {
        const auto &c = r.candidates[i];
        napi_value item = nullptr;
        napi_create_object(env, &item);
        set(env, item, "width", number(env, c.width));
        set(env, item, "height", number(env, c.height));
        set(env, item, "scale", number(env, c.scale));
        set(env, item, "score", number(env, c.score));
        set(env, item, "exact", boolean(env, c.kind == PP_CANDIDATE_EXACT));
        napi_set_element(env, candidates, static_cast<uint32_t>(i), item);
    }
    set(env, out, "candidates", candidates);
    return out;
}

napi_value init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"analyze", nullptr, analyze, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, 1, desc);
    return exports;
}

} // namespace

static napi_module pixelpeeper_module = {
    1,
    0,
    nullptr,
    init,
    "pixelpeeper",
    nullptr,
    {0},
};

extern "C" __attribute__((constructor)) void RegisterPixelPeeperModule(void) {
    napi_module_register(&pixelpeeper_module);
}
