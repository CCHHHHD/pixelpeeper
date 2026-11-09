/*
 * PixelPeeper native core.
 *
 * This is a dependency-free C++17 port of pixelpeeper/analysis.py.  The
 * detector is intentionally kept numerically close to the Python reference:
 * the same candidate ladder, phase search, tile voting and confidence rules
 * are used.  Only the evidence-image bookkeeping (recon curves, tile map,
 * duplicate-gap samples) is omitted from the C ABI.
 */
#include "pixelpeeper.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace {

struct Image {
    int w = 0;
    int h = 0;
    std::vector<float> pix;
    float at(int x, int y) const { return pix[static_cast<size_t>(y) * w + x]; }
};

struct Profile {
    std::vector<double> d1;
    std::vector<double> d2;
};

struct Evidence {
    pp_method method = PP_METHOD_NONE;
    double scale = 1.0;
    int period = 0;
    double duty = 0.0;
    double match = 0.0;
    int support = 0;
    double spread = 0.0;
    int tiles = 0;
    double prominence = 0.0;
    double phase = 0.0;
    bool near_native = false;
};

struct DupCand {
    double scale = 1.0;
    int period = 0;
    double duty = 0.0;
    double match = 0.0;
};

struct Curves {
    std::vector<std::array<double, 3>> linear;
    std::vector<std::array<double, 3>> nearest;
};

struct AxisChoice {
    Evidence ev;
    Curves curves;
};

// ---------------------------------------------------------------------------
// small numeric helpers (NumPy-compatible defaults)
// ---------------------------------------------------------------------------
double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    std::sort(v.begin(), v.end());
    if (n & 1u) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = (static_cast<double>(v.size()) - 1.0) * p / 100.0;
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return v[lo];
    const double frac = pos - static_cast<double>(lo);
    return v[lo] + (v[hi] - v[lo]) * frac;
}

double mean_of(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double stddev_pop(const float *p, size_t n) {
    if (n == 0) return 0.0;
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += p[i];
    mean /= static_cast<double>(n);
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(p[i]) - mean;
        var += d * d;
    }
    return std::sqrt(var / static_cast<double>(n));
}

// np.linspace(...).astype(int), i.e. truncation toward zero.
std::vector<int> linspace_int(int count, int n) {
    std::vector<int> out;
    if (count <= 0 || n <= 0) return out;
    out.reserve(static_cast<size_t>(count));
    if (count == 1) {
        out.push_back(0);
        return out;
    }
    const double step = static_cast<double>(n - 1) / static_cast<double>(count - 1);
    for (int i = 0; i < count; ++i) out.push_back(static_cast<int>(static_cast<double>(i) * step));
    return out;
}

// ---------------------------------------------------------------------------
// image preparation
// ---------------------------------------------------------------------------
Image luma_from_rgb(const uint8_t *rgb, int w, int h, int stride) {
    Image im;
    im.w = w;
    im.h = h;
    im.pix.resize(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        const uint8_t *row = rgb + static_cast<size_t>(y) * stride;
        float *dst = im.pix.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const float r = row[x * 3 + 0];
            const float g = row[x * 3 + 1];
            const float b = row[x * 3 + 2];
            dst[x] = (0.2126f * r + 0.7152f * g + 0.0722f * b) / 255.0f;
        }
    }
    return im;
}

Image luma_from_rgba(const uint8_t *rgba, int w, int h, int stride) {
    Image im;
    im.w = w;
    im.h = h;
    im.pix.resize(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        const uint8_t *row = rgba + static_cast<size_t>(y) * stride;
        float *dst = im.pix.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const float r = row[x * 4 + 0];
            const float g = row[x * 4 + 1];
            const float b = row[x * 4 + 2];
            dst[x] = (0.2126f * r + 0.7152f * g + 0.0722f * b) / 255.0f;
        }
    }
    return im;
}

Image luma_from_plane(const uint8_t *p, int w, int h, int stride) {
    Image im;
    im.w = w;
    im.h = h;
    im.pix.resize(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        const uint8_t *row = p + static_cast<size_t>(y) * stride;
        float *dst = im.pix.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) dst[x] = static_cast<float>(row[x]) / 255.0f;
    }
    return im;
}

Image crop_image(const Image &src, int x0, int y0, int x1, int y1) {
    x0 = std::max(0, std::min(x0, src.w - 1));
    y0 = std::max(0, std::min(y0, src.h - 1));
    x1 = std::max(x0 + 1, std::min(x1, src.w));
    y1 = std::max(y0 + 1, std::min(y1, src.h));
    Image out;
    out.w = x1 - x0;
    out.h = y1 - y0;
    out.pix.resize(static_cast<size_t>(out.w) * out.h);
    for (int y = 0; y < out.h; ++y) {
        const float *srcrow = src.pix.data() + static_cast<size_t>(y + y0) * src.w + x0;
        std::copy(srcrow, srcrow + out.w, out.pix.data() + static_cast<size_t>(y) * out.w);
    }
    return out;
}

std::vector<int> detect_bars(const Image &im) {
    const int h = im.h;
    const int w = im.w;
    const double lim = 9.0 / 255.0;

    auto row_stats = [&](int y, double &mean, double &sd) {
        const float *row = im.pix.data() + static_cast<size_t>(y) * w;
        mean = 0.0;
        for (int x = 0; x < w; ++x) mean += row[x];
        mean /= static_cast<double>(w);
        double var = 0.0;
        for (int x = 0; x < w; ++x) {
            const double d = static_cast<double>(row[x]) - mean;
            var += d * d;
        }
        sd = std::sqrt(var / static_cast<double>(w));
    };
    auto col_stats = [&](int x, double &mean, double &sd) {
        mean = 0.0;
        for (int y = 0; y < h; ++y) mean += im.at(x, y);
        mean /= static_cast<double>(h);
        double var = 0.0;
        for (int y = 0; y < h; ++y) {
            const double d = static_cast<double>(im.at(x, y)) - mean;
            var += d * d;
        }
        sd = std::sqrt(var / static_cast<double>(h));
    };
    auto run = [&](bool rows, int n, bool reverse) {
        int k = 0;
        while (k < n - 2) {
            const int idx = reverse ? (n - 1 - k) : k;
            double m, s;
            if (rows) row_stats(idx, m, s);
            else col_stats(idx, m, s);
            if (!(m < lim && s < 0.008)) break;
            ++k;
        }
        return k;
    };

    int top = run(true, h, false);
    int bottom = run(true, h, true);
    int left = run(false, w, false);
    int right = run(false, w, true);

    if (!(top > 0 && bottom > 0)) top = bottom = 0;
    if (!(left > 0 && right > 0)) left = right = 0;
    if (top + bottom > h * 0.45) top = bottom = 0;
    if (left + right > w * 0.45) left = right = 0;
    return {top, bottom, left, right};
}

Profile profiles(const Image &im, int axis) {
    // axis == 1: per-column derivatives; axis == 0: per-row derivatives.
    Profile p;
    if (axis == 1) {
        if (im.w < 2) return p;
        p.d1.assign(static_cast<size_t>(im.w - 1), 0.0);
        for (int y = 0; y < im.h; ++y) {
            const float *row = im.pix.data() + static_cast<size_t>(y) * im.w;
            for (int x = 0; x + 1 < im.w; ++x) p.d1[x] += std::fabs(static_cast<double>(row[x + 1]) - row[x]);
        }
        for (double &v : p.d1) v /= static_cast<double>(im.h);
        if (im.w >= 3) {
            p.d2.assign(static_cast<size_t>(im.w - 2), 0.0);
            for (int y = 0; y < im.h; ++y) {
                const float *row = im.pix.data() + static_cast<size_t>(y) * im.w;
                for (int x = 0; x + 2 < im.w; ++x) {
                    p.d2[x] += std::fabs(static_cast<double>(row[x + 2]) - 2.0 * row[x + 1] + row[x]);
                }
            }
            for (double &v : p.d2) v /= static_cast<double>(im.h);
        }
    } else {
        if (im.h < 2) return p;
        p.d1.assign(static_cast<size_t>(im.h - 1), 0.0);
        for (int x = 0; x < im.w; ++x) {
            for (int y = 0; y + 1 < im.h; ++y) {
                p.d1[y] += std::fabs(static_cast<double>(im.at(x, y + 1)) - im.at(x, y));
            }
        }
        for (double &v : p.d1) v /= static_cast<double>(im.w);
        if (im.h >= 3) {
            p.d2.assign(static_cast<size_t>(im.h - 2), 0.0);
            for (int x = 0; x < im.w; ++x) {
                for (int y = 0; y + 2 < im.h; ++y) {
                    p.d2[y] += std::fabs(static_cast<double>(im.at(x, y + 2)) -
                                         2.0 * im.at(x, y + 1) + im.at(x, y));
                }
            }
            for (double &v : p.d2) v /= static_cast<double>(im.w);
        }
    }
    return p;
}

// ---------------------------------------------------------------------------
// detector 1: duplicate pixels / nearest-neighbour upscaling
// ---------------------------------------------------------------------------
struct PeriodResult {
    double match = 0.0;
    int period = 0;
    double duty = 0.0;
};

PeriodResult best_period(const std::vector<uint8_t> &mask, int pmin, int pmax) {
    struct Scored {
        double match;
        int period;
        double duty;
    };
    std::vector<Scored> scored;
    const int n = static_cast<int>(mask.size());
    double ones_total = 0.0;
    for (uint8_t v : mask) ones_total += v;
    for (int p = pmin; p <= pmax; ++p) {
        std::vector<double> cnt(static_cast<size_t>(p), 0.0);
        std::vector<double> ones(static_cast<size_t>(p), 0.0);
        for (int i = 0; i < n; ++i) {
            const int r = i % p;
            cnt[r] += 1.0;
            ones[r] += mask[i];
        }
        std::vector<uint8_t> pred(static_cast<size_t>(n), 0);
        int equal = 0;
        for (int i = 0; i < n; ++i) {
            const int r = i % p;
            const bool bit = (ones[r] / std::max(1.0, cnt[r])) >= 0.5;
            pred[i] = bit ? 1 : 0;
            if (pred[i] == mask[i]) ++equal;
        }
        scored.push_back({static_cast<double>(equal) / std::max(1, n), p,
                          ones_total / std::max(1, n)});
    }
    if (scored.empty()) return {};
    double top = -1.0;
    for (const auto &s : scored) top = std::max(top, s.match);
    std::sort(scored.begin(), scored.end(), [](const Scored &a, const Scored &b) {
        return a.period < b.period;
    });
    for (const auto &s : scored) {
        if (s.match >= top - 0.004) return {s.match, s.period, s.duty};
    }
    return {scored.front().match, scored.front().period, scored.front().duty};
}

std::vector<DupCand> dup_search(const std::vector<double> &profile,
                                int max_period = 16,
                                double min_match = 0.992,
                                double min_duty = 0.06,
                                double max_duty = 0.74) {
    std::vector<DupCand> out;
    const int n = static_cast<int>(profile.size());
    if (n < 48) return out;
    for (double v : profile) {
        if (!std::isfinite(v)) return out;
    }
    const double hi = percentile(profile, 72.0);
    std::vector<double> uniq;
    uniq.reserve(profile.size());
    for (double v : profile) {
        if (v <= hi) uniq.push_back(v);
    }
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    if (uniq.empty()) return out;
    if (uniq.size() > 40) {
        std::vector<double> sampled;
        for (int idx : linspace_int(34, static_cast<int>(uniq.size()))) sampled.push_back(uniq[idx]);
        uniq.swap(sampled);
    }
    std::vector<double> thrs;
    const size_t n24 = std::min<size_t>(24, uniq.size());
    for (size_t i = 0; i < n24; ++i) thrs.push_back(uniq[i]);
    const size_t n18 = std::min<size_t>(18, uniq.size());
    for (size_t i = 0; i < n18; ++i) {
        thrs.push_back(std::nextafter(uniq[i], std::numeric_limits<double>::infinity()));
    }
    std::sort(thrs.begin(), thrs.end());
    thrs.erase(std::unique(thrs.begin(), thrs.end()), thrs.end());

    std::vector<DupCand> found;
    for (double t : thrs) {
        std::vector<uint8_t> mask(static_cast<size_t>(n), 0);
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            if (profile[i] <= t) {
                mask[i] = 1;
                sum += 1.0;
            }
        }
        const double duty0 = sum / static_cast<double>(n);
        if (duty0 < min_duty || duty0 > max_duty) continue;
        const PeriodResult pr = best_period(mask, 2, max_period);
        if (pr.match >= min_match && pr.duty >= min_duty && pr.duty <= max_duty) {
            int lows = static_cast<int>(std::llround(pr.period * pr.duty));
            lows = std::max(1, std::min(pr.period - 1, lows));
            found.push_back({static_cast<double>(pr.period) /
                                 static_cast<double>(pr.period - lows),
                             pr.period, pr.duty, pr.match});
        }
    }
    if (found.empty()) return out;
    std::sort(found.begin(), found.end(), [](const DupCand &a, const DupCand &b) {
        if (a.match != b.match) return a.match > b.match;
        return a.scale < b.scale;
    });
    std::vector<std::vector<DupCand>> groups;
    for (const DupCand &c : found) {
        bool placed = false;
        for (auto &g : groups) {
            if (g.front().period == c.period && std::fabs(g.front().scale - c.scale) < 0.03) {
                g.push_back(c);
                placed = true;
                break;
            }
        }
        if (!placed) groups.push_back({c});
    }
    std::sort(groups.begin(), groups.end(), [](const std::vector<DupCand> &a,
                                               const std::vector<DupCand> &b) {
        double ma = 0.0, mb = 0.0;
        for (const auto &c : a) ma = std::max(ma, c.match);
        for (const auto &c : b) mb = std::max(mb, c.match);
        if (ma != mb) return ma > mb;
        return a.size() > b.size();
    });
    const size_t keep = std::min<size_t>(4, groups.size());
    for (size_t i = 0; i < keep; ++i) {
        const auto &g = groups[i];
        std::vector<double> duties;
        double m = 0.0;
        for (const auto &c : g) {
            duties.push_back(c.duty);
            m = std::max(m, c.match);
        }
        const double duty = median_of(duties);
        const int p = g.front().period;
        int lows = static_cast<int>(std::llround(p * duty));
        lows = std::max(1, std::min(p - 1, lows));
        out.push_back({static_cast<double>(p) / static_cast<double>(p - lows), p, duty, m});
    }
    return out;
}

struct Vote {
    double scale;
    double match;
    double texture;
};

std::vector<Vote> nearest_tiles(const Image &im, int axis, int max_period = 16) {
    std::vector<Vote> votes;
    const int H = axis == 1 ? im.h : im.w;
    const int W = axis == 1 ? im.w : im.h;
    int size = std::max(160, std::min(384, std::min(H, W) / 3));
    std::vector<std::pair<int, int>> grid;
    if (H <= size || W <= size) {
        size = std::min(H, W);
        grid.push_back({0, 0});
    } else {
        const int step_y = std::max(1, (H - size) / 2);
        const int step_x = std::max(1, (W - size) / 4);
        for (int y = 0; y <= H - size; y += step_y) {
            for (int x = 0; x <= W - size; x += step_x) grid.push_back({y, x});
        }
    }
    for (const auto &pos : grid) {
        const int y0 = pos.first;
        const int x0 = pos.second;
        Image t;
        t.w = size;
        t.h = size;
        t.pix.resize(static_cast<size_t>(size) * size);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                t.pix[static_cast<size_t>(y) * size + x] =
                    axis == 1 ? im.at(x0 + x, y0 + y) : im.at(y0 + y, x0 + x);
            }
        }
        const double tex = stddev_pop(t.pix.data(), t.pix.size());
        if (tex < 0.035) continue;
        const Profile p = profiles(t, 1);
        const auto cands = dup_search(p.d1, max_period);
        if (cands.empty()) continue;
        const DupCand &c = cands.front();
        double dmin = std::numeric_limits<double>::infinity();
        for (double v : p.d1) dmin = std::min(dmin, v);
        const double p90 = percentile(p.d1, 90.0);
        const double cut = dmin + 0.02 * std::max(1e-9, p90);
        std::vector<double> low;
        for (double v : p.d1) {
            if (v <= cut) low.push_back(v);
        }
        if (low.empty()) continue;
        if (median_of(low) > 0.04 * std::max(1e-9, p90)) continue;
        votes.push_back({c.scale, c.match, tex});
    }
    return votes;
}

double dup_phase(const std::vector<double> &d1, int period) {
    if (period < 2) return 0.0;
    const double thr = std::max(1e-6, percentile(d1, 90.0) * 0.02);
    std::vector<int> counts(static_cast<size_t>(period), 0);
    for (size_t i = 0; i < d1.size(); ++i) {
        if (d1[i] <= thr) counts[i % static_cast<size_t>(period)] += 1;
    }
    int best = 0;
    for (int i = 1; i < period; ++i) {
        if (counts[i] > counts[best]) best = i;
    }
    return static_cast<double>(best);
}

Evidence nearest_axis(const Image &im, int axis) {
    const std::string name = axis == 1 ? "x" : "y";
    (void)name;
    std::vector<Vote> votes = nearest_tiles(im, axis);
    const Profile global = profiles(im, axis);
    const auto cands = dup_search(global.d1);
    const DupCand *c = cands.empty() ? nullptr : &cands.front();

    if (c) {
        // A real duplicate grid alternates columns/rows that are identical
        // across all rows with ones that are not.
        if (c->duty > 0.02 && c->duty < 0.95) {
            const double q = percentile(global.d1, c->duty * 100.0);
            std::vector<double> eq;
            eq.reserve(global.d1.size());
            if (axis == 1) {
                for (int x = 0; x + 1 < im.w; ++x) {
                    int same = 0;
                    for (int y = 0; y < im.h; ++y) {
                        if (im.at(x + 1, y) == im.at(x, y)) ++same;
                    }
                    eq.push_back(static_cast<double>(same) / im.h);
                }
            } else {
                for (int y = 0; y + 1 < im.h; ++y) {
                    int same = 0;
                    for (int x = 0; x < im.w; ++x) {
                        if (im.at(x, y + 1) == im.at(x, y)) ++same;
                    }
                    eq.push_back(static_cast<double>(same) / im.w);
                }
            }
            std::vector<double> a, b;
            bool any = false, all = true;
            for (size_t i = 0; i < global.d1.size(); ++i) {
                if (global.d1[i] <= q) {
                    a.push_back(eq[i]);
                    any = true;
                } else {
                    b.push_back(eq[i]);
                    all = false;
                }
            }
            if (any && !all) {
                const double contrast = median_of(a) - median_of(b);
                if (contrast < 0.25) return Evidence{};
            }
        }
    }
    if (c) votes.push_back({c->scale, c->match, 1.0});
    if (votes.empty()) return Evidence{};

    std::vector<double> scales, matches;
    for (const auto &v : votes) {
        scales.push_back(v.scale);
        matches.push_back(v.match);
    }
    double med = median_of(scales);
    std::vector<double> close_scales, close_matches;
    for (size_t i = 0; i < scales.size(); ++i) {
        if (std::fabs(scales[i] - med) <= std::max(0.02, 0.012 * med)) {
            close_scales.push_back(scales[i]);
            close_matches.push_back(matches[i]);
        }
    }
    if (close_scales.size() >= 2) med = median_of(close_scales);
    std::vector<double> dev;
    for (double s : close_scales) dev.push_back(std::fabs(s - med));
    const double spread = close_scales.empty() ? 0.0 : median_of(dev);

    Evidence ev;
    ev.method = PP_METHOD_NEAREST;
    ev.scale = med;
    ev.period = c ? c->period : 0;
    ev.duty = c ? c->duty : 0.0;
    ev.match = close_matches.empty() ? 0.0 : median_of(close_matches);
    ev.support = static_cast<int>(close_scales.size());
    ev.spread = spread;
    ev.tiles = static_cast<int>(votes.size());
    ev.phase = c ? dup_phase(global.d1, c->period) : 0.0;
    return ev;
}

// ---------------------------------------------------------------------------
// detector 2: resampling fit
// ---------------------------------------------------------------------------
const std::vector<double> &candidate_scales() {
    static const std::vector<double> scales = [] {
        std::vector<double> s;
        for (double v : {1.050, 1.055, 1.060, 1.065, 1.070, 1.075, 1.080, 1.085,
                         1.090, 1.095, 1.100, 1.105, 1.110, 1.115, 1.120, 1.125,
                         1.130, 1.135, 1.140, 1.145, 1.150, 1.155, 1.160, 1.165,
                         1.170, 1.175, 1.180, 1.185, 1.190, 1.195, 1.200,
                         1.800, 1.805, 1.810, 1.815, 1.820, 1.825, 1.830, 1.835,
                         1.840, 1.845, 1.850, 1.855, 1.860, 1.865, 1.870, 1.875,
                         1.880, 1.885, 1.890, 1.895, 1.900, 1.905, 1.910, 1.915,
                         1.920, 1.925, 1.930, 1.935, 1.940, 1.945, 1.950, 1.955,
                         1.960, 1.965, 1.970, 1.975, 1.980, 1.985, 1.990, 1.995,
                         2.000, 2.005, 2.010, 2.015, 2.020, 2.025, 2.030, 2.035,
                         2.040, 2.045, 2.050,
                         1.222, 1.250, 1.286, 1.300, 1.333, 1.375, 1.400, 1.429,
                         1.500, 1.600, 1.625, 1.667, 1.714, 1.750, 2.100, 2.200,
                         2.250, 2.400, 2.500, 2.667, 2.800, 3.000, 3.200, 3.500,
                         4.000}) {
            s.push_back(v);
        }
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
        return s;
    }();
    return scales;
}

std::pair<double, double> recon_pair(const std::vector<std::vector<float>> &rows,
                                     double s, double phi) {
    const int nrows = static_cast<int>(rows.size());
    const int W = nrows ? static_cast<int>(rows.front().size()) : 0;
    if (nrows == 0 || W < 2) return {1.0, 1.0};
    const int k = static_cast<int>((static_cast<double>(W) - 1.0 - phi) / s) + 1;
    if (k < 12) return {1.0, 1.0};

    std::vector<float> xs(static_cast<size_t>(k));
    std::vector<int> i0(static_cast<size_t>(k));
    std::vector<float> f(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        xs[i] = static_cast<float>(phi + static_cast<double>(i) * s);
        i0[i] = std::max(0, std::min(W - 2, static_cast<int>(std::floor(xs[i]))));
        f[i] = xs[i] - static_cast<float>(i0[i]);
    }

    std::vector<float> t(static_cast<size_t>(W));
    std::vector<int> j(static_cast<size_t>(W));
    std::vector<float> w(static_cast<size_t>(W));
    for (int x = 0; x < W; ++x) {
        t[x] = static_cast<float>((static_cast<double>(x) + 0.5 - phi) / s);
        j[x] = std::max(0, std::min(k - 2, static_cast<int>(std::floor(t[x]))));
        w[x] = t[x] - static_cast<float>(j[x]);
    }

    std::vector<float> lival(static_cast<size_t>(k));
    std::vector<double> per_row(static_cast<size_t>(nrows), 0.0);
    for (int r = 0; r < nrows; ++r) {
        const auto &row = rows[r];
        for (int i = 0; i < k; ++i) {
            const float lo = row[i0[i]];
            lival[i] = lo + (row[i0[i] + 1] - lo) * f[i];
        }
        double sum = 0.0;
        for (int x = 0; x < W; ++x) {
            const float a = lival[j[x]];
            const float rec = a + (lival[j[x] + 1] - a) * w[x];
            const double d = static_cast<double>(row[x]) - rec;
            sum += d * d;
        }
        per_row[r] = sum / W;
    }
    std::vector<double> lin = per_row;

    // nearest-neighbour reconstruction uses the same source samples
    for (int r = 0; r < nrows; ++r) {
        const auto &row = rows[r];
        for (int i = 0; i < k; ++i) {
            const float lo = row[i0[i]];
            lival[i] = lo + (row[i0[i] + 1] - lo) * f[i];
        }
        double sum = 0.0;
        for (int x = 0; x < W; ++x) {
            const double d = static_cast<double>(row[x]) - lival[j[x]];
            sum += d * d;
        }
        per_row[r] = sum / W;
    }
    std::vector<double> near = per_row;

    auto trimmed = [](std::vector<double> v) {
        if (v.size() >= 6) {
            const size_t keep = std::max<size_t>(4, static_cast<size_t>(v.size() * 0.7));
            std::sort(v.begin(), v.end());
            v.resize(keep);
        }
        return mean_of(v);
    };
    return {trimmed(lin), trimmed(near)};
}

struct PairResult {
    double lin = 1e30;
    double lin_phi = 0.0;
    double near = 1e30;
    double near_phi = 0.0;
};

PairResult best_pair(const std::vector<std::vector<float>> &rows, double s, int phases) {
    PairResult best;
    for (int i = 0; i < phases; ++i) {
        const double phi = static_cast<double>(i) * s / phases;
        const auto e = recon_pair(rows, s, phi);
        if (e.first < best.lin) {
            best.lin = e.first;
            best.lin_phi = phi;
        }
        if (e.second < best.near) {
            best.near = e.second;
            best.near_phi = phi;
        }
    }
    return best;
}

std::vector<std::vector<float>> select_rows(const Image &im, int axis, int count = 40) {
    const int H = axis == 1 ? im.h : im.w;
    const int W = axis == 1 ? im.w : im.h;
    const std::vector<int> idx = linspace_int(std::min(count, H), H);
    std::vector<std::vector<float>> rows;
    for (int i : idx) {
        std::vector<float> row(static_cast<size_t>(W));
        for (int x = 0; x < W; ++x) row[x] = axis == 1 ? im.at(x, i) : im.at(i, x);
        rows.push_back(std::move(row));
    }
    if (!rows.empty()) {
        std::vector<std::vector<float>> kept;
        for (const auto &row : rows) {
            const double sd = stddev_pop(row.data(), row.size());
            double grad = 0.0;
            for (size_t x = 0; x + 1 < row.size(); ++x) grad += std::fabs(row[x + 1] - row[x]);
            if (!row.empty()) grad /= static_cast<double>(row.size());
            if (sd > 0.02 && grad > 0.004) kept.push_back(row);
        }
        if (kept.size() >= 8) return kept;
    }
    return rows;
}

AxisChoice recon_axis(const Image &im, int axis) {
    AxisChoice choice;
    choice.ev.method = PP_METHOD_NONE;
    const auto rows = select_rows(im, axis);
    if (rows.empty() || rows.front().size() < 200 || rows.size() < 6) return choice;

    double sq = 0.0;
    size_t cnt = 0;
    for (const auto &row : rows) {
        for (float v : row) {
            sq += static_cast<double>(v) * v;
            ++cnt;
        }
    }
    const double base = (cnt ? sq / static_cast<double>(cnt) : 1.0) > 1e-12
                            ? sq / static_cast<double>(cnt) : 1.0;

    Curves curves;
    for (double s : candidate_scales()) {
        const PairResult p = best_pair(rows, s, 14);
        const PairResult lo = best_pair(rows, s - 0.10, 6);
        const PairResult hi = best_pair(rows, s + 0.10, 6);
        const double prom_lin = (std::min(lo.lin, hi.lin) - p.lin) / std::max(p.lin, 1e-12);
        const double prom_near = (std::min(lo.near, hi.near) - p.near) / std::max(p.near, 1e-12);
        curves.linear.push_back({s, p.lin / base, prom_lin});
        curves.nearest.push_back({s, p.near / base, prom_near});
    }

    struct Pick {
        double s, e, prom;
        int kernel; // 0 linear, 1 nearest
    };
    std::vector<Pick> picks;
    const double ceilings[2] = {0.0035, 0.0025};
    const double thresholds[2] = {0.10, 0.18};
    for (int k = 0; k < 2; ++k) {
        const auto &curve = k == 0 ? curves.linear : curves.nearest;
        bool have = false;
        Pick best{0, 1e30, 0, k};
        for (const auto &c : curve) {
            if (c[2] >= thresholds[k] && c[1] <= ceilings[k]) {
                if (!have || c[1] < best.e) {
                    best = {c[0], c[1], c[2], k};
                    have = true;
                }
            }
        }
        if (have) picks.push_back(best);
    }
    if (picks.empty()) return choice;

    Pick pick{};
    if (picks.size() == 2) {
        const Pick &lin = picks[0];
        const Pick &nr = picks[1];
        if (std::fabs(lin.s - nr.s) <= 0.03) {
            pick = lin;
        } else if (nr.prom > 2.0 * lin.prom && nr.e < lin.e) {
            pick = nr;
        } else {
            pick = lin.e <= nr.e ? lin : nr;
        }
    } else {
        pick = picks.front();
    }

    double s = pick.s;
    double e = pick.e;
    double prom = pick.prom;
    int kernel = pick.kernel;
    bool near_native = false;

    double gmin = 1e30;
    for (const auto &c : curves.linear) gmin = std::min(gmin, c[1]);
    for (const auto &c : curves.nearest) gmin = std::min(gmin, c[1]);
    std::vector<Pick> near;
    for (int k = 0; k < 2; ++k) {
        const auto &curve = k == 0 ? curves.linear : curves.nearest;
        for (const auto &c : curve) {
            if (c[0] >= 1.055 && c[0] <= 1.105 && c[2] >= 0.12 && c[1] <= 3.0 * gmin) {
                near.push_back({c[0], c[1], c[2], k});
            }
        }
    }
    if (!near.empty() && ((s >= 1.12 && s <= 1.35) || (s >= 1.9 && prom < 0.5))) {
        std::sort(near.begin(), near.end(), [](const Pick &a, const Pick &b) {
            if (a.s != b.s) return a.s < b.s;
            return a.e < b.e;
        });
        s = near.front().s;
        e = near.front().e;
        prom = near.front().prom;
        kernel = near.front().kernel;
        near_native = true;
    }
    (void)e;

    const PairResult fine = best_pair(rows, s, 16);
    Evidence ev;
    ev.method = kernel == 1 ? PP_METHOD_NEAREST : PP_METHOD_SMOOTH;
    ev.scale = s;
    ev.prominence = prom;
    ev.phase = kernel == 1 ? fine.near_phi : fine.lin_phi;
    ev.match = std::min(0.99, 0.75 + prom);
    ev.near_native = near_native;
    choice.ev = ev;
    choice.curves = std::move(curves);
    return choice;
}

// ---------------------------------------------------------------------------
// candidate ladder and full-frame analysis
// ---------------------------------------------------------------------------
struct LadderEntry {
    int w;
    int h;
};

std::vector<LadderEntry> ladder_for(double aspect) {
    std::vector<LadderEntry> out = {
        {640, 360}, {854, 480}, {960, 540}, {1024, 576}, {1152, 648},
        {1280, 720}, {1366, 768}, {1440, 810}, {1600, 900}, {1680, 945},
        {1920, 1080}, {2133, 1200}, {2560, 1440}, {2880, 1620},
        {3200, 1800}, {3840, 2160}, {5120, 2880}, {7680, 4320}};
    if (std::fabs(aspect - 21.0 / 9.0) < 0.15) {
        out.insert(out.end(), {{2560, 1080}, {3440, 1440}, {3840, 1600}, {5120, 2160}});
    }
    if (std::fabs(aspect - 16.0 / 10.0) < 0.08) {
        out.insert(out.end(), {{1280, 800}, {1440, 900}, {1680, 1050},
                               {1920, 1200}, {2560, 1600}, {3840, 2400}});
    }
    return out;
}

std::vector<pp_candidate> resolution_candidates(int w, int h, double sx, double sy) {
    const double iw = static_cast<double>(w) / sx;
    const double ih = static_cast<double>(h) / sy;
    const double target_ar = static_cast<double>(w) / h;
    std::vector<pp_candidate> out;
    std::set<std::pair<int, int>> seen;
    for (const auto &rung : ladder_for(target_ar)) {
        const int rw = rung.w, rh = rung.h;
        const double ar = static_cast<double>(rw) / rh;
        if (std::fabs(ar - target_ar) / target_ar > 0.05) continue;
        const double s = (static_cast<double>(w) / rw + static_cast<double>(h) / rh) / 2.0;
        if (s > 9.0 || s < 1.02) continue;
        const double err = std::fabs(std::log(s / ((sx + sy) / 2.0)));
        if (err > 0.09) continue;
        double common = 1.0;
        if (rh == 720 || rh == 1080 || rh == 1440 || rh == 2160 || rh == 900 ||
            rh == 540 || rh == 810 || rh == 1200 || rh == 1800 || rh == 1620) {
            common = 1.2;
        }
        if (rh == 810 || rh == 900 || rh == 540 || rh == 720 || rh == 1080) common += 0.1;
        const double score = std::exp(-std::pow(err / 0.030, 2.0)) * common;
        if (seen.count({rw, rh})) continue;
        seen.insert({rw, rh});
        pp_candidate c{};
        c.width = rw;
        c.height = rh;
        c.scale = s;
        c.score = score;
        c.kind = PP_CANDIDATE_COMMON;
        std::snprintf(c.label, sizeof(c.label), "%dx%d", rw, rh);
        out.push_back(c);
    }
    const int ew = static_cast<int>(std::llround(iw));
    const int eh = static_cast<int>(std::llround(ih));
    if (ew > 0 && eh > 0) {
        pp_candidate c{};
        c.width = ew;
        c.height = eh;
        c.scale = (sx + sy) / 2.0;
        c.score = 0.5;
        c.kind = PP_CANDIDATE_EXACT;
        std::snprintf(c.label, sizeof(c.label), "%dx%d", ew, eh);
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(), [](const pp_candidate &a, const pp_candidate &b) {
        return a.score > b.score;
    });
    std::vector<pp_candidate> unique;
    for (const auto &c : out) {
        bool dup = false;
        for (const auto &u : unique) {
            if (std::abs(c.width - u.width) <= 2 && std::abs(c.height - u.height) <= 2) {
                dup = true;
                break;
            }
        }
        if (!dup) unique.push_back(c);
    }
    if (unique.size() > PP_MAX_CANDIDATES) unique.resize(PP_MAX_CANDIDATES);
    return unique;
}

double block_variance_2x(const Image &im) {
    const int h = (im.h / 2) * 2;
    const int w = (im.w / 2) * 2;
    if (h < 16 || w < 16) return 1.0;
    double within_sum = 0.0;
    const long long blocks = static_cast<long long>(h / 2) * (w / 2);
    for (int by = 0; by < h; by += 2) {
        for (int bx = 0; bx < w; bx += 2) {
            const double a = im.at(bx, by), b = im.at(bx + 1, by),
                         c = im.at(bx, by + 1), d = im.at(bx + 1, by + 1);
            const double m = (a + b + c + d) / 4.0;
            within_sum += ((a - m) * (a - m) + (b - m) * (b - m) +
                           (c - m) * (c - m) + (d - m) * (d - m)) / 4.0;
        }
    }
    const double within = within_sum / static_cast<double>(blocks);
    double mean = 0.0;
    for (float v : im.pix) mean += v;
    mean /= static_cast<double>(im.pix.size());
    double total = 0.0;
    for (float v : im.pix) {
        const double d = static_cast<double>(v) - mean;
        total += d * d;
    }
    total /= static_cast<double>(im.pix.size());
    return total > 1e-12 ? within / total : 1.0;
}

void analyse_core(const Image &input, int base_w, int base_h, const pp_roi *roi,
                  bool quick, int depth, pp_result &res);

pp_result unwrap_2x(const Image &content, const pp_result &outer, bool quick, int depth) {
    pp_result best;
    double best_score = -1.0;
    bool have = false;
    for (int pass = 0; pass < 2; ++pass) {
        const int oy = pass, ox = pass;
        Image half;
        half.w = (content.w - ox) / 2;
        half.h = (content.h - oy) / 2;
        if (half.w < 2 || half.h < 2) continue;
        half.pix.resize(static_cast<size_t>(half.w) * half.h);
        for (int y = 0; y < half.h; ++y) {
            for (int x = 0; x < half.w; ++x) {
                half.pix[static_cast<size_t>(y) * half.w + x] = content.at(ox + 2 * x, oy + 2 * y);
            }
        }
        pp_result inner;
        pp_result_init(&inner);
        analyse_core(half, half.w, half.h, nullptr, quick, depth + 1, inner);
        const double score = inner.confidence + (inner.method != PP_METHOD_NONE ? 0.25 : 0.0);
        if (!have || score > best_score) {
            best = inner;
            best_score = score;
            have = true;
        }
    }
    if (!have) return best;

    const bool combine = best.method != PP_METHOD_NONE && std::fabs(best.scale - 2.0) >= 0.06;
    const double inner_scale = combine ? best.scale : 1.0;
    const double total = 2.0 * inner_scale;

    best.width = content.w;
    best.height = content.h;
    best.content_box[0] = 0;
    best.content_box[1] = 0;
    best.content_box[2] = content.w;
    best.content_box[3] = content.h;
    best.bars[0] = best.bars[1] = best.bars[2] = best.bars[3] = 0;
    best.has_roi = 0;
    std::memset(&best.roi, 0, sizeof(best.roi));
    best.scale = total;
    best.scale_x = 2.0 * (combine ? best.scale_x : 1.0);
    best.scale_y = 2.0 * (combine ? best.scale_y : 1.0);
    best.internal_w = static_cast<int>(std::llround(content.w / best.scale_x));
    best.internal_h = static_cast<int>(std::llround(content.h / best.scale_y));
    best.percent = 100.0 / total;
    if (combine) {
        best.confidence = clampd(best.confidence * 0.95, 0.4, 0.97);
    } else {
        best.method = (outer.method == PP_METHOD_NEAREST &&
                       std::fabs(outer.scale - 2.0) < 0.06)
                          ? PP_METHOD_NEAREST
                          : PP_METHOD_SMOOTH;
        best.confidence = std::max(0.70, outer.confidence);
    }
    if (combine) {
        if (best.evidence_x.method != PP_METHOD_NONE) best.evidence_x.scale *= 2.0;
        if (best.evidence_y.method != PP_METHOD_NONE) best.evidence_y.scale *= 2.0;
    }
    const auto cands = resolution_candidates(content.w, content.h, best.scale_x, best.scale_y);
    best.candidate_count = static_cast<int>(cands.size());
    for (int i = 0; i < best.candidate_count; ++i) best.candidates[i] = cands[i];
    return best;
}

void analyse_core(const Image &input, int base_w, int base_h, const pp_roi *roi,
                  bool quick, int depth, pp_result &res) {
    pp_result_init(&res);
    res.width = base_w;
    res.height = base_h;
    res.has_roi = roi ? 1 : 0;
    if (roi) res.roi = *roi;

    const std::vector<int> bars = detect_bars(input);
    const int top = bars[0], bottom = bars[1], left = bars[2], right = bars[3];
    for (int i = 0; i < 4; ++i) res.bars[i] = bars[i];
    const int cw = input.w - left - right;
    const int ch = input.h - top - bottom;
    if (roi) {
        res.content_box[0] = roi->x0;
        res.content_box[1] = roi->y0;
        res.content_box[2] = roi->x1;
        res.content_box[3] = roi->y1;
    } else {
        res.content_box[0] = left;
        res.content_box[1] = top;
        res.content_box[2] = left + cw;
        res.content_box[3] = top + ch;
    }
    const Image content = crop_image(input, left, top, left + cw, top + ch);
    if (content.w < 2 || content.h < 2) {
        std::snprintf(res.error, sizeof(res.error), "image is too small after cropping");
        return;
    }

    // Ignore the outer 5%: HUD and letterbox remnants tend to be native-res.
    const int mx = static_cast<int>(content.w * 0.05);
    const int my = static_cast<int>(content.h * 0.05);
    const Image work = (content.w > 60 && content.h > 60)
                           ? crop_image(content, mx, my, content.w - mx, content.h - my)
                           : content;

    const Evidence evx = nearest_axis(work, 1);
    const Evidence evy = nearest_axis(work, 0);
    AxisChoice rx = recon_axis(work, 1);
    AxisChoice ry = recon_axis(work, 0);

    const bool nearest_ok_x = evx.method == PP_METHOD_NEAREST && evx.support >= 2 && evx.match >= 0.995;
    const bool nearest_ok_y = evy.method == PP_METHOD_NEAREST && evy.support >= 2 && evy.match >= 0.995;
    const bool recon_ok_x = (rx.ev.method == PP_METHOD_NEAREST && rx.ev.prominence >= 0.18) ||
                            (rx.ev.method == PP_METHOD_SMOOTH && rx.ev.prominence >= 0.10);
    const bool recon_ok_y = (ry.ev.method == PP_METHOD_NEAREST && ry.ev.prominence >= 0.18) ||
                            (ry.ev.method == PP_METHOD_SMOOTH && ry.ev.prominence >= 0.10);

    pp_method method = PP_METHOD_NONE;
    bool near_used = false;
    double scale_x = 1.0, scale_y = 1.0;
    if (nearest_ok_x || nearest_ok_y) {
        method = PP_METHOD_NEAREST;
        if (nearest_ok_x && nearest_ok_y) {
            scale_x = evx.scale;
            scale_y = evy.scale;
        } else if (nearest_ok_x) {
            scale_x = scale_y = evx.scale;
        } else {
            scale_x = scale_y = evy.scale;
        }
    } else if (recon_ok_x || recon_ok_y) {
        near_used = rx.ev.near_native || ry.ev.near_native;
        if (near_used) {
            const Evidence &win = rx.ev.near_native ? rx.ev : ry.ev;
            scale_x = scale_y = win.scale;
            method = win.method;
        } else if (recon_ok_x && recon_ok_y) {
            if (std::fabs(rx.ev.scale - ry.ev.scale) /
                    std::max(rx.ev.scale, ry.ev.scale) > 0.03) {
                const Evidence *win = rx.ev.prominence >= ry.ev.prominence ? &rx.ev : &ry.ev;
                if (std::min(rx.ev.prominence, ry.ev.prominence) >=
                    0.4 * std::max(rx.ev.prominence, ry.ev.prominence)) {
                    win = rx.ev.scale < ry.ev.scale ? &rx.ev : &ry.ev;
                }
                scale_x = scale_y = win->scale;
                method = win->method;
            } else {
                scale_x = scale_y = (rx.ev.scale + ry.ev.scale) / 2.0;
                method = (rx.ev.method == PP_METHOD_NEAREST ||
                          ry.ev.method == PP_METHOD_NEAREST)
                             ? PP_METHOD_NEAREST
                             : PP_METHOD_SMOOTH;
            }
        } else if (recon_ok_x) {
            scale_x = scale_y = rx.ev.scale;
            method = rx.ev.method;
        } else {
            scale_x = scale_y = ry.ev.scale;
            method = ry.ev.method;
        }
    } else {
        res.native_likely = 1;
    }

    if (method != PP_METHOD_NONE) {
        res.method = method;
        res.scale_x = clampd(scale_x, 1.0, 10.0);
        res.scale_y = clampd(scale_y, 1.0, 10.0);
        res.scale = clampd((res.scale_x + res.scale_y) / 2.0, 1.0, 10.0);
        res.internal_w = static_cast<int>(std::llround(base_w / res.scale_x));
        res.internal_h = static_cast<int>(std::llround(base_h / res.scale_y));
        res.percent = 100.0 / res.scale;
        if (method == PP_METHOD_NEAREST) {
            if (nearest_ok_x || nearest_ok_y) {
                res.evidence_x = evx.method != PP_METHOD_NONE
                                     ? pp_evidence{evx.method, evx.scale, evx.period, evx.duty,
                                                   evx.match, evx.support, evx.spread, evx.tiles,
                                                   evx.prominence, evx.phase,
                                                   evx.near_native ? 1 : 0}
                                     : pp_evidence{PP_METHOD_NEAREST, res.scale_x, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                res.evidence_y = evy.method != PP_METHOD_NONE
                                     ? pp_evidence{evy.method, evy.scale, evy.period, evy.duty,
                                                   evy.match, evy.support, evy.spread, evy.tiles,
                                                   evy.prominence, evy.phase,
                                                   evy.near_native ? 1 : 0}
                                     : pp_evidence{PP_METHOD_NEAREST, res.scale_y, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                const int support = (nearest_ok_x && nearest_ok_y)
                                        ? std::min(evx.support, evy.support)
                                        : std::max(evx.support, evy.support);
                res.confidence = clampd(0.88 + 0.02 * std::min(support, 5), 0.0, 0.97);
                if (support < 2) res.confidence = 0.72;
            } else {
                res.evidence_x = rx.ev.method != PP_METHOD_NONE
                                     ? pp_evidence{rx.ev.method, rx.ev.scale, 0, 0, rx.ev.match,
                                                   0, 0, 0, rx.ev.prominence, rx.ev.phase, 0}
                                     : pp_evidence{PP_METHOD_NEAREST, res.scale_x, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                res.evidence_y = ry.ev.method != PP_METHOD_NONE
                                     ? pp_evidence{ry.ev.method, ry.ev.scale, 0, 0, ry.ev.match,
                                                   0, 0, 0, ry.ev.prominence, ry.ev.phase, 0}
                                     : pp_evidence{PP_METHOD_NEAREST, res.scale_y, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                const double prom = std::max(rx.ev.method != PP_METHOD_NONE ? rx.ev.prominence : 0.0,
                                             ry.ev.method != PP_METHOD_NONE ? ry.ev.prominence : 0.0);
                res.confidence = clampd(0.78 + 0.18 * std::min(prom, 1.0), 0.45, 0.96);
            }
            if (near_used) res.confidence = std::min(res.confidence, 0.55);
        } else {
            res.evidence_x = rx.ev.method != PP_METHOD_NONE
                                 ? pp_evidence{PP_METHOD_SMOOTH, res.scale_x, 0, 0, rx.ev.match,
                                               0, 0, 0, rx.ev.prominence, rx.ev.phase,
                                               rx.ev.near_native ? 1 : 0}
                                 : pp_evidence{PP_METHOD_SMOOTH, res.scale_x, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            res.evidence_y = ry.ev.method != PP_METHOD_NONE
                                 ? pp_evidence{PP_METHOD_SMOOTH, res.scale_y, 0, 0, ry.ev.match,
                                               0, 0, 0, ry.ev.prominence, ry.ev.phase,
                                               ry.ev.near_native ? 1 : 0}
                                 : pp_evidence{PP_METHOD_SMOOTH, res.scale_y, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            const double prom = std::max(rx.ev.prominence, ry.ev.prominence);
            res.confidence = clampd(0.58 + 0.55 * std::min(prom, 0.7), 0.35, 0.88);
            if (near_used) res.confidence = std::min(res.confidence, 0.55);
        }
        if (res.percent < 20.0 || res.scale > 4.5) res.confidence *= 0.85;
        if (res.scale < 1.20) res.confidence = std::min(res.confidence, 0.55);
    } else {
        res.scale = 1.0;
        res.internal_w = base_w;
        res.internal_h = base_h;
        res.percent = 100.0;
        res.confidence = 0.55;
        if (evx.method != PP_METHOD_NONE) {
            res.evidence_x = {evx.method, evx.scale, evx.period, evx.duty, evx.match,
                              evx.support, evx.spread, evx.tiles, evx.prominence,
                              evx.phase, evx.near_native ? 1 : 0};
        } else if (rx.ev.method != PP_METHOD_NONE) {
            res.evidence_x = {rx.ev.method, rx.ev.scale, 0, 0, rx.ev.match, 0, 0, 0,
                              rx.ev.prominence, rx.ev.phase, 0};
        }
        if (evy.method != PP_METHOD_NONE) {
            res.evidence_y = {evy.method, evy.scale, evy.period, evy.duty, evy.match,
                              evy.support, evy.spread, evy.tiles, evy.prominence,
                              evy.phase, evy.near_native ? 1 : 0};
        } else if (ry.ev.method != PP_METHOD_NONE) {
            res.evidence_y = {ry.ev.method, ry.ev.scale, 0, 0, ry.ev.match, 0, 0, 0,
                              ry.ev.prominence, ry.ev.phase, 0};
        }
    }

    const auto cands = resolution_candidates(base_w, base_h, res.scale_x, res.scale_y);
    res.candidate_count = static_cast<int>(cands.size());
    for (int i = 0; i < res.candidate_count; ++i) res.candidates[i] = cands[i];

    // OS/compositor 2x unwrap, only for whole-frame images.
    if (depth == 0 && roi == nullptr && base_w == input.w && base_h == input.h) {
        const double bv = block_variance_2x(content);
        const bool suspicious = bv < 0.045 &&
                                (res.method == PP_METHOD_NONE ||
                                 (res.method == PP_METHOD_NEAREST &&
                                  std::fabs(res.scale - 2.0) < 0.06));
        if (suspicious) {
            pp_result combined = unwrap_2x(content, res, quick, depth);
            if (combined.confidence > 0.0) {
                res = combined;
                return;
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// public C API
// ---------------------------------------------------------------------------
extern "C" {

void pp_result_init(pp_result *out) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    out->scale_x = 1.0;
    out->scale_y = 1.0;
    out->scale = 1.0;
    out->percent = 100.0;
    out->method = PP_METHOD_NONE;
    out->evidence_x.method = PP_METHOD_NONE;
    out->evidence_y.method = PP_METHOD_NONE;
    out->evidence_x.scale = 1.0;
    out->evidence_y.scale = 1.0;
}

const char *pp_version(void) { return PP_VERSION_STRING; }

int pp_analyze_rgb(const uint8_t *rgb, int width, int height, int stride_bytes,
                   const pp_roi *roi, int quick, pp_result *out) {
    if (!out) return -1;
    pp_result_init(out);
    const auto started = std::chrono::steady_clock::now();
    auto finish = [&]() {
        out->elapsed_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
    };
    if (!rgb || width < 2 || height < 2 || stride_bytes < width * 3) {
        std::snprintf(out->error, sizeof(out->error), "invalid RGB buffer or dimensions");
        finish();
        return -1;
    }
    pp_roi r{};
    int base_w = width, base_h = height;
    Image full;
    if (roi) {
        r = *roi;
        r.x0 = std::max(0, std::min(r.x0, width - 2));
        r.y0 = std::max(0, std::min(r.y0, height - 2));
        r.x1 = std::max(r.x0 + 2, std::min(r.x1, width));
        r.y1 = std::max(r.y0 + 2, std::min(r.y1, height));
        full = luma_from_rgb(rgb, width, height, stride_bytes);
        full = crop_image(full, r.x0, r.y0, r.x1, r.y1);
    } else {
        full = luma_from_rgb(rgb, width, height, stride_bytes);
    }
    if (full.w < 2 || full.h < 2) {
        std::snprintf(out->error, sizeof(out->error), "image is too small");
        finish();
        return -1;
    }
    analyse_core(full, base_w, base_h, roi ? &r : nullptr, quick != 0, 0, *out);
    finish();
    return 0;
}

int pp_analyze_rgba(const uint8_t *rgba, int width, int height, int stride_bytes,
                    const pp_roi *roi, int quick, pp_result *out) {
    if (!out) return -1;
    pp_result_init(out);
    const auto started = std::chrono::steady_clock::now();
    auto finish = [&]() {
        out->elapsed_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
    };
    if (!rgba || width < 2 || height < 2 || stride_bytes < width * 4) {
        std::snprintf(out->error, sizeof(out->error), "invalid RGBA buffer or dimensions");
        finish();
        return -1;
    }
    pp_roi r{};
    int base_w = width, base_h = height;
    Image full;
    if (roi) {
        r = *roi;
        r.x0 = std::max(0, std::min(r.x0, width - 2));
        r.y0 = std::max(0, std::min(r.y0, height - 2));
        r.x1 = std::max(r.x0 + 2, std::min(r.x1, width));
        r.y1 = std::max(r.y0 + 2, std::min(r.y1, height));
        full = luma_from_rgba(rgba, width, height, stride_bytes);
        full = crop_image(full, r.x0, r.y0, r.x1, r.y1);
    } else {
        full = luma_from_rgba(rgba, width, height, stride_bytes);
    }
    if (full.w < 2 || full.h < 2) {
        std::snprintf(out->error, sizeof(out->error), "image is too small");
        finish();
        return -1;
    }
    analyse_core(full, base_w, base_h, roi ? &r : nullptr, quick != 0, 0, *out);
    finish();
    return 0;
}

int pp_analyze_luma(const uint8_t *luma, int width, int height, int stride_bytes,
                    const pp_roi *roi, int quick, pp_result *out) {
    if (!out) return -1;
    pp_result_init(out);
    const auto started = std::chrono::steady_clock::now();
    auto finish = [&]() {
        out->elapsed_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
    };
    if (!luma || width < 2 || height < 2 || stride_bytes < width) {
        std::snprintf(out->error, sizeof(out->error), "invalid luma buffer or dimensions");
        finish();
        return -1;
    }
    pp_roi r{};
    int base_w = width, base_h = height;
    Image full;
    if (roi) {
        r = *roi;
        r.x0 = std::max(0, std::min(r.x0, width - 2));
        r.y0 = std::max(0, std::min(r.y0, height - 2));
        r.x1 = std::max(r.x0 + 2, std::min(r.x1, width));
        r.y1 = std::max(r.y0 + 2, std::min(r.y1, height));
        full = luma_from_plane(luma, width, height, stride_bytes);
        full = crop_image(full, r.x0, r.y0, r.x1, r.y1);
    } else {
        full = luma_from_plane(luma, width, height, stride_bytes);
    }
    if (full.w < 2 || full.h < 2) {
        std::snprintf(out->error, sizeof(out->error), "image is too small");
        finish();
        return -1;
    }
    analyse_core(full, base_w, base_h, roi ? &r : nullptr, quick != 0, 0, *out);
    finish();
    return 0;
}

} // extern "C"
