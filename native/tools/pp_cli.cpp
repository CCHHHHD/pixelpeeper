// Small command-line driver used by the regression tests.
// Usage: pp_cli image.rgb WIDTH HEIGHT [--quick] [--roi x0,y0,x1,y1]
#include "pixelpeeper.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s image.rgb WIDTH HEIGHT [--quick] [--roi x0,y0,x1,y1]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const int w = std::atoi(argv[2]);
    const int h = std::atoi(argv[3]);
    bool quick = false;
    pp_roi roi{};
    bool has_roi = false;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) {
            quick = true;
        } else if (std::strcmp(argv[i], "--roi") == 0 && i + 1 < argc) {
            int x0, y0, x1, y1;
            if (std::sscanf(argv[++i], "%d,%d,%d,%d", &x0, &y0, &x1, &y1) == 4) {
                roi = {x0, y0, x1, y1};
                has_roi = true;
            }
        }
    }
    if (w <= 0 || h <= 0) {
        std::fprintf(stderr, "invalid dimensions\n");
        return 2;
    }
    const size_t need = static_cast<size_t>(w) * h * 3;
    std::vector<unsigned char> data(need);
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::perror(path);
        return 2;
    }
    const size_t got = std::fread(data.data(), 1, need, f);
    std::fclose(f);
    if (got != need) {
        std::fprintf(stderr, "short read: got %zu of %zu bytes\n", got, need);
        return 2;
    }
    pp_result r;
    const int rc = pp_analyze_rgb(data.data(), w, h, w * 3, has_roi ? &roi : nullptr,
                                  quick ? 1 : 0, &r);
    if (rc != 0) {
        std::fprintf(stderr, "analysis failed: %s\n", r.error);
        return 1;
    }
    std::printf("{\n");
    std::printf("  \"width\": %d, \"height\": %d,\n", r.width, r.height);
    std::printf("  \"content_box\": [%d, %d, %d, %d],\n", r.content_box[0], r.content_box[1],
                r.content_box[2], r.content_box[3]);
    std::printf("  \"bars\": [%d, %d, %d, %d],\n", r.bars[0], r.bars[1], r.bars[2], r.bars[3]);
    std::printf("  \"has_roi\": %s, \"roi\": [%d, %d, %d, %d],\n", r.has_roi ? "true" : "false",
                r.roi.x0, r.roi.y0, r.roi.x1, r.roi.y1);
    std::printf("  \"scale_x\": %.12g, \"scale_y\": %.12g, \"scale\": %.12g,\n",
                r.scale_x, r.scale_y, r.scale);
    std::printf("  \"internal_w\": %d, \"internal_h\": %d, \"percent\": %.12g,\n",
                r.internal_w, r.internal_h, r.percent);
    const char *method = r.method == PP_METHOD_NEAREST ? "nearest"
                         : r.method == PP_METHOD_SMOOTH ? "smooth" : "none";
    std::printf("  \"method\": \"%s\", \"confidence\": %.12g, \"native_likely\": %s,\n",
                method, r.confidence, r.native_likely ? "true" : "false");
    auto evidence = [](const char *name, const pp_evidence &e) {
        const char *m = e.method == PP_METHOD_NEAREST ? "nearest"
                        : e.method == PP_METHOD_SMOOTH ? "smooth" : "none";
        std::printf("  \"%s\": {\"method\": \"%s\", \"scale\": %.12g, \"period\": %d, "
                    "\"duty\": %.12g, \"match\": %.12g, \"support\": %d, \"spread\": %.12g, "
                    "\"tiles\": %d, \"prominence\": %.12g, \"phase\": %.12g, "
                    "\"near_native\": %s}",
                    name, m, e.scale, e.period, e.duty, e.match, e.support, e.spread,
                    e.tiles, e.prominence, e.phase, e.near_native ? "true" : "false");
    };
    evidence("evidence_x", r.evidence_x);
    std::printf(",\n");
    evidence("evidence_y", r.evidence_y);
    std::printf(",\n  \"candidates\": [");
    for (int i = 0; i < r.candidate_count; ++i) {
        const auto &c = r.candidates[i];
        std::printf("%s{\"width\": %d, \"height\": %d, \"scale\": %.12g, \"score\": %.12g, "
                    "\"kind\": \"%s\", \"label\": \"%s\"}",
                    i ? ", " : "", c.width, c.height, c.scale, c.score,
                    c.kind == PP_CANDIDATE_EXACT ? "exact" : "common", c.label);
    }
    std::printf("],\n  \"elapsed_ms\": %.6f\n}\n", r.elapsed_ms);
    return 0;
}
