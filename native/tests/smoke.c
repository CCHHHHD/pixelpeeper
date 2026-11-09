/* Plain-C smoke test: proves the public header is consumable from C. */
#include "pixelpeeper.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    unsigned char rgb[96 * 96 * 3];
    unsigned char rgba[96 * 96 * 4];
    unsigned char luma[96 * 96];
    for (int y = 0; y < 96; ++y) {
        for (int x = 0; x < 96; ++x) {
            const int i = y * 96 + x;
            rgb[i * 3 + 0] = (unsigned char)(x * 2);
            rgb[i * 3 + 1] = (unsigned char)(y * 2);
            rgb[i * 3 + 2] = (unsigned char)((x + y) & 0xff);
            rgba[i * 4 + 0] = rgb[i * 3 + 0];
            rgba[i * 4 + 1] = rgb[i * 3 + 1];
            rgba[i * 4 + 2] = rgb[i * 3 + 2];
            rgba[i * 4 + 3] = 255;
            luma[i] = (unsigned char)((x * 3 + y * 5) & 0xff);
        }
    }

    pp_result r;
    pp_result_init(&r);
    if (r.scale != 1.0 || r.method != PP_METHOD_NONE) {
        fprintf(stderr, "pp_result_init failed\n");
        return 1;
    }
    if (strcmp(pp_version(), PP_VERSION_STRING) != 0) {
        fprintf(stderr, "version mismatch\n");
        return 1;
    }
    if (pp_analyze_rgb(rgb, 96, 96, 96 * 3, NULL, 1, &r) != 0) {
        fprintf(stderr, "RGB analysis failed: %s\n", r.error);
        return 1;
    }
    if (pp_analyze_luma(luma, 96, 96, 96, NULL, 1, &r) != 0) {
        fprintf(stderr, "luma analysis failed: %s\n", r.error);
        return 1;
    }
    if (pp_analyze_rgba(rgba, 96, 96, 96 * 4, NULL, 1, &r) != 0) {
        fprintf(stderr, "RGBA analysis failed: %s\n", r.error);
        return 1;
    }
    {
        pp_result rgb_result, rgba_result;
        if (pp_analyze_rgb(rgb, 96, 96, 96 * 3, NULL, 1, &rgb_result) != 0 ||
            pp_analyze_rgba(rgba, 96, 96, 96 * 4, NULL, 1, &rgba_result) != 0 ||
            rgb_result.method != rgba_result.method ||
            rgb_result.internal_w != rgba_result.internal_w ||
            rgb_result.internal_h != rgba_result.internal_h ||
            rgb_result.scale != rgba_result.scale) {
            fprintf(stderr, "RGB and RGBA paths disagree\n");
            return 1;
        }
    }

    pp_roi roi = {8, 8, 80, 80};
    if (pp_analyze_rgb(rgb, 96, 96, 96 * 3, &roi, 1, &r) != 0 || !r.has_roi) {
        fprintf(stderr, "ROI analysis failed\n");
        return 1;
    }

    if (pp_analyze_rgb(NULL, 96, 96, 96 * 3, NULL, 1, &r) != -1) {
        fprintf(stderr, "NULL buffer was accepted\n");
        return 1;
    }
    if (r.error[0] == '\0') {
        fprintf(stderr, "error string was not set\n");
        return 1;
    }
    printf("pixelpeeper C API %s OK (%d x %d)\n", pp_version(), r.width, r.height);
    return 0;
}
