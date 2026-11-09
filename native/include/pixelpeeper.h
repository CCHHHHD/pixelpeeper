/*
 * PixelPeeper native C API
 *
 * Portable C ABI for the PixelPeeper screenshot-resolution detector.
 * The library has no external dependencies and accepts an 8-bit RGB buffer,
 * which makes it usable from Android NDK/JNI and HarmonyOS Native/NAPI.
 *
 * Build: see native/README.md
 * License: MIT
 */
#ifndef PIXELPEEPER_H
#define PIXELPEEPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PP_VERSION_MAJOR 1
#define PP_VERSION_MINOR 0
#define PP_VERSION_PATCH 0
#define PP_VERSION_STRING "1.0.0"

#if defined(_WIN32)
#  if defined(PP_BUILD_SHARED)
#    define PP_API __declspec(dllexport)
#  elif defined(PP_USE_SHARED)
#    define PP_API __declspec(dllimport)
#  else
#    define PP_API
#  endif
#else
#  if defined(PP_BUILD_SHARED) && defined(__GNUC__)
#    define PP_API __attribute__((visibility("default")))
#  else
#    define PP_API
#  endif
#endif

/* Maximum number of ranked resolution suggestions kept in a result. */
#define PP_MAX_CANDIDATES 8
#define PP_LABEL_CAPACITY 40
#define PP_ERROR_CAPACITY 160

typedef enum pp_method {
    PP_METHOD_NONE = 0,   /* no stable grid found (native / temporal upscaler) */
    PP_METHOD_NEAREST = 1,/* nearest-neighbour / integer point upscale */
    PP_METHOD_SMOOTH = 2  /* bilinear / bicubic / FSR-style upscale */
} pp_method;

typedef enum pp_candidate_kind {
    PP_CANDIDATE_COMMON = 0,
    PP_CANDIDATE_EXACT = 1
} pp_candidate_kind;

/* Pixel rectangle, half-open: [x0, x1) and [y0, y1). */
typedef struct pp_roi {
    int x0;
    int y0;
    int x1;
    int y1;
} pp_roi;

typedef struct pp_evidence {
    pp_method method;
    double scale;       /* measured upscale factor on this axis */
    int period;         /* repeat period in output pixels (nearest only) */
    double duty;        /* fraction of duplicated positions */
    double match;       /* how well the period explains the mask (0..1) */
    int support;        /* number of tiles that voted for this scale */
    double spread;      /* median absolute deviation of the tile votes */
    int tiles;          /* tiles with enough texture */
    double prominence;  /* depth of the resampling-fit dip */
    double phase;       /* source-grid phase, in output pixels */
    int near_native;    /* credible <1.2x grid */
} pp_evidence;

typedef struct pp_candidate {
    int width;
    int height;
    double scale;
    double score;
    pp_candidate_kind kind;
    char label[PP_LABEL_CAPACITY]; /* UTF-8, NUL-terminated */
} pp_candidate;

typedef struct pp_result {
    int width;              /* full screenshot width */
    int height;             /* full screenshot height */
    int content_box[4];     /* x0, y0, x1, y1 after letterbox removal */
    int bars[4];            /* top, bottom, left, right black bars */
    int has_roi;
    pp_roi roi;             /* valid when has_roi != 0 */

    double scale_x;
    double scale_y;
    double scale;
    int internal_w;
    int internal_h;
    double percent;         /* 100 / scale */

    pp_method method;
    double confidence;      /* 0..1 */
    int native_likely;
    pp_evidence evidence_x;
    pp_evidence evidence_y;

    pp_candidate candidates[PP_MAX_CANDIDATES];
    int candidate_count;

    double elapsed_ms;
    char error[PP_ERROR_CAPACITY]; /* empty on success */
} pp_result;

/*
 * Analyse an interleaved RGB8 image.
 *
 * rgb          pointer to the first byte of the top-left pixel
 * width,height image dimensions in pixels, both >= 2
 * stride_bytes distance between row starts; must be >= width * 3.
 *              Pass width * 3 for tightly packed data.
 * roi          optional region to analyse, in full-image coordinates;
 *              NULL analyses the whole image. The reported resolution is
 *              always relative to the full image.
 * quick        reserved compatibility flag (0 or 1). The C ABI does not
 *              expose the per-tile dynamic-resolution heat map, so this does
 *              not change the returned resolution; it is kept for parity with
 *              the Python API and future extensions.
 * out          caller-owned result; initialised by this function.
 *
 * Returns 0 on success, -1 on invalid arguments, -2 if the analysis failed.
 * The function never allocates memory that the caller must release and is
 * safe to call from multiple threads with separate result objects.
 */
PP_API int pp_analyze_rgb(const uint8_t *rgb,
                          int width,
                          int height,
                          int stride_bytes,
                          const pp_roi *roi,
                          int quick,
                          pp_result *out);

/*
 * Same as pp_analyze_rgb, but for interleaved RGBA8888 data.  Alpha is
 * ignored.  This is the native layout used by Android Bitmap (ARGB_8888 in
 * memory) and by HarmonyOS PixelMap (RGBA_8888).
 */
PP_API int pp_analyze_rgba(const uint8_t *rgba,
                           int width,
                           int height,
                           int stride_bytes,
                           const pp_roi *roi,
                           int quick,
                           pp_result *out);

/*
 * Same as pp_analyze_rgb, but takes a tightly packed 8-bit luma plane.
 * Useful when the platform decoder already produced YUV/NV12 and only the
 * Y plane is needed.
 */
PP_API int pp_analyze_luma(const uint8_t *luma,
                           int width,
                           int height,
                           int stride_bytes,
                           const pp_roi *roi,
                           int quick,
                           pp_result *out);

/* Returns "1.0.0" (static storage, never freed). */
PP_API const char *pp_version(void);

/* Fills *out with a safe zeroed result. out may be NULL. */
PP_API void pp_result_init(pp_result *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PIXELPEEPER_H */
