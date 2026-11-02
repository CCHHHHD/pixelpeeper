"""Pixel-grid analysis: infer a game's internal render resolution from a screenshot.

A screenshot is saved at the display resolution, while a game often renders at a
lower internal resolution and lets the GPU/display scale the frame up.  That
resampling leaves a regular grid in the pixel data:

* nearest-neighbour scaling duplicates whole pixels/rows -> exact zeros in the
  first derivative;
* bilinear/bicubic scaling is piecewise polynomial on the source grid -> a
  periodic pattern in the higher derivatives.

Two independent detectors are used:

1. *duplicate-pixel detector* — measures what fraction of columns/rows are
   exact duplicates.  For a scale ``s`` that fraction is ``1 - 1/s``, so
   ``s = 1 / (1 - fraction)``.  Very precise for pixel-art / nearest-neighbour
   style upscales (including fractional factors such as 1.5x).
2. *resampling-fit detector* — samples the image on a candidate source grid,
   reconstructs it by (bi)linear interpolation and measures the residual.  At
   the true grid the residual has a sharp dip; this works for bilinear, bicubic
   and most "soft" upscalers, where no pixels are literally duplicated.

The result is reported with a confidence and, when the evidence is ambiguous,
with a ranked list of plausible internal resolutions instead of pretending to
know the exact pixel count.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps

__all__ = [
    "AnalysisResult",
    "AxisEvidence",
    "ResolutionCandidate",
    "analyse_image",
    "analyse_array",
    "load_image",
]


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class AxisEvidence:
    axis: str = "x"                    # "x" or "y"
    method: str = "none"               # "nearest", "smooth" or "none"
    scale: float = 1.0
    period: int = 0                    # repeat period in output pixels
    duty: float = 0.0                  # fraction of "inside a source pixel" positions
    match: float = 0.0                 # how well the period explains the mask (0-1)
    support: int = 0                   # number of tiles that voted for this scale
    spread: float = 0.0                # median absolute deviation of the tile votes
    tiles: int = 0                     # tiles with enough texture
    prominence: float = 0.0            # depth of the resampling-fit dip
    phase: float = 0.0                 # phase of the source grid, in pixels
    near_native: bool = False          # evidence for a <1.2x, near-native grid
    note: str = ""


@dataclass
class ResolutionCandidate:
    width: int
    height: int
    scale: float
    score: float
    label: str
    kind: str = "common"               # "common" or "exact"


@dataclass
class AnalysisResult:
    width: int
    height: int
    content_box: tuple[int, int, int, int]
    bars: tuple[int, int, int, int] = (0, 0, 0, 0)
    path: str = ""
    roi: tuple[int, int, int, int] | None = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale: float = 1.0
    internal_w: int = 0
    internal_h: int = 0
    percent: float = 100.0
    method: str = "none"               # "nearest" | "smooth" | "none"
    confidence: float = 0.0
    native_likely: bool = False
    candidates: list[ResolutionCandidate] = field(default_factory=list)
    evidence_x: AxisEvidence = field(default_factory=AxisEvidence)
    evidence_y: AxisEvidence = field(default_factory=AxisEvidence)
    tile_map: list[list[float]] = field(default_factory=list)
    tile_grid: tuple[int, int] = (0, 0)
    duplicate_gaps: list[int] = field(default_factory=list)
    pattern: list[float] = field(default_factory=list)
    recon_x: list[tuple[float, float, float]] = field(default_factory=list)
    recon_y: list[tuple[float, float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def has_grid(self) -> bool:
        return self.method in ("nearest", "smooth") and self.scale >= 1.02

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def summary(self) -> str:
        if self.method == "none":
            if self.native_likely:
                return f"原生 {self.width} × {self.height}（未发现放大网格）"
            return "无法确定（未发现稳定的像素网格）"
        kind = "像素级放大" if self.method == "nearest" else "平滑放大"
        return f"≈ {self.internal_w} × {self.internal_h}（{self.percent:.1f}% · {self.scale:.3f}×，{kind}）"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _luma(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32)
    if a.ndim == 2:
        return a / 255.0
    return (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]) / 255.0


def _moving_average(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or a.size < 2:
        return a.copy()
    win = min(win, a.size)
    k = np.ones(win, np.float64) / win
    pad = win // 2
    p = np.pad(a.astype(np.float64), (pad, win - 1 - pad), mode="edge")
    return np.convolve(p, k, mode="valid")


def _best_period(mask: np.ndarray, periods: Iterable[int]) -> tuple[float, int, float]:
    """Return (match, period, duty), preferring the fundamental period."""
    n = mask.size
    x = np.arange(n)
    scored: list[tuple[float, int, float]] = []
    for p in periods:
        idx = x % p
        cnt = np.bincount(idx, minlength=p)
        ones = np.bincount(idx, weights=mask, minlength=p)
        frac = ones / np.maximum(cnt, 1)
        pred = frac[idx] >= 0.5
        scored.append((float((pred == mask).mean()), p, float(mask.mean())))
    if not scored:
        return 0.0, 0, 0.0
    top = max(m for m, _, _ in scored)
    for m, p, d in sorted(scored, key=lambda t: t[1]):
        if m >= top - 0.004:
            return m, p, d
    return scored[0]


# --------------------------------------------------------------------------- #
# detector 1: duplicate pixels (nearest-neighbour upscaling)
# --------------------------------------------------------------------------- #
@dataclass
class _DupCand:
    scale: float
    period: int
    duty: float
    match: float


def _dup_search(profile: np.ndarray, *, max_period: int = 16,
                min_match: float = 0.992, min_duty: float = 0.06,
                max_duty: float = 0.74) -> list[_DupCand]:
    v = np.asarray(profile, np.float64)
    if v.size < 48 or not np.isfinite(v).all():
        return []
    hi = float(np.percentile(v, 72))
    uniq = np.unique(v[v <= hi])
    if uniq.size == 0:
        return []
    if uniq.size > 40:
        uniq = uniq[np.unique(np.linspace(0, uniq.size - 1, 34).astype(int))]
    thrs = [float(t) for t in uniq[:24]] + [float(np.nextafter(t, np.inf)) for t in uniq[:18]]
    found: list[_DupCand] = []
    for t in sorted(set(thrs)):
        mask = v <= t
        duty0 = float(mask.mean())
        if duty0 < min_duty or duty0 > max_duty:
            continue
        match, period, duty = _best_period(mask, range(2, max_period + 1))
        if match >= min_match and min_duty <= duty <= max_duty:
            # Use integer counts per period: a 1.5x upscale has period 3 with
            # one duplicate position, i.e. scale = 3 / (3 - 1).  This removes
            # the small bias of a global fraction estimate.
            lows = int(round(period * duty))
            lows = max(1, min(period - 1, lows))
            found.append(_DupCand(period / (period - lows), period, duty, match))
    if not found:
        return []
    found.sort(key=lambda c: (-c.match, c.scale))
    groups: list[list[_DupCand]] = []
    for c in found:
        for g in groups:
            if g[0].period == c.period and abs(g[0].scale - c.scale) < 0.03:
                g.append(c)
                break
        else:
            groups.append([c])
    groups.sort(key=lambda g: (-max(c.match for c in g), -len(g)))
    out = []
    for g in groups[:4]:
        duty = float(np.median([c.duty for c in g]))
        p = g[0].period
        lows = max(1, min(p - 1, int(round(p * duty))))
        out.append(_DupCand(p / (p - lows), p, duty, max(c.match for c in g)))
    return out


def _profiles(gray: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    g = gray if axis == 1 else gray.T
    d1 = np.abs(np.diff(g, axis=1)).mean(axis=0)
    d2 = np.abs(np.diff(g, n=2, axis=1)).mean(axis=0)
    return d1.astype(np.float64), d2.astype(np.float64)


def _nearest_tiles(gray: np.ndarray, axis: int, max_period: int = 16) -> list[tuple[float, float, float]]:
    """Tile votes (scale, quality, texture) for the duplicate-pixel detector."""
    g = gray if axis == 1 else gray.T
    H, W = g.shape
    size = int(max(160, min(384, min(H, W) // 3)))
    if H <= size or W <= size:
        size = min(H, W)
        grid = [(0, 0)]
    else:
        # keep the tile count modest (max 5x3): enough for a vote, fast on 4K
        step_y = max(1, (H - size) // 2)
        step_x = max(1, (W - size) // 4)
        grid = [(y, x) for y in range(0, H - size + 1, step_y)
                for x in range(0, W - size + 1, step_x)]
    out: list[tuple[float, float, float]] = []
    for y, x in grid:
        t = g[y:y + size, x:x + size]
        tex = float(t.std())
        if tex < 0.035:
            continue
        d1, _ = _profiles(t, 1)
        cands = _dup_search(d1, max_period=max_period)
        if not cands:
            continue
        c = cands[0]
        low = d1[d1 <= (d1.min() + 0.02 * max(1e-9, np.percentile(d1, 90)))]
        if low.size == 0:
            continue
        # a real nearest-neighbour duplicate cluster sits at the very bottom
        if np.median(low) > 0.04 * max(1e-9, np.percentile(d1, 90)):
            continue
        out.append((c.scale, c.match, tex))
    return out


def _nearest_axis(gray: np.ndarray, axis: int) -> AxisEvidence:
    votes = _nearest_tiles(gray, axis)
    min_d1, d2 = _profiles(gray, axis)
    cands = _dup_search(min_d1)
    c = cands[0] if cands else None
    name = "x" if axis == 1 else "y"
    if c:
        g = gray if axis == 1 else gray.T
        # A real duplicate grid alternates between columns/rows that are
        # identical across *all* rows and ones that are not.  Flat drawings
        # contain long identical runs too, so require that contrast.
        eq = (np.diff(g, axis=1) == 0).mean(axis=0)
        if 0.02 < c.duty < 0.95:
            thr = float(np.quantile(min_d1, c.duty))
            mask = min_d1 <= thr
            if mask.any() and (~mask).any():
                contrast = float(np.median(eq[mask]) - np.median(eq[~mask]))
                if contrast < 0.25:
                    return AxisEvidence(name, "none")
    if c:
        votes.append((c.scale, c.match, 1.0))
    if not votes:
        return AxisEvidence(name, "none")
    scales = np.array([v[0] for v in votes])
    matches = np.array([v[1] for v in votes])
    med = float(np.median(scales))
    close = np.abs(scales - med) <= max(0.02, 0.012 * med)
    if close.sum() >= 2:
        med = float(np.median(scales[close]))
    spread = float(np.median(np.abs(scales[close] - med))) if close.any() else 0.0
    support = int(close.sum())
    if support < 3 and len(votes) > 2:
        # too few agreeing tiles: keep it, but the caller will lower confidence
        pass
    return AxisEvidence(
        axis=name, method="nearest", scale=med,
        period=c.period if c else 0,
        duty=c.duty if c else 0.0,
        match=float(np.median(matches[close])) if close.any() else 0.0,
        support=support, spread=spread, tiles=len(votes),
        phase=_dup_phase(min_d1, c.period if c else 0),
    )


def _dup_phase(d1: np.ndarray, period: int) -> float:
    if period < 2:
        return 0.0
    thr = max(1e-6, float(np.percentile(d1, 90)) * 0.02)
    pos = np.flatnonzero(d1 <= thr)
    if pos.size == 0:
        return 0.0
    counts = np.bincount(pos % period, minlength=period)
    return float(np.argmax(counts))


# --------------------------------------------------------------------------- #
# detector 2: resampling fit (bilinear / bicubic / soft upscalers)
# --------------------------------------------------------------------------- #
# Plausible scale factors: standard 16:9 heights, common resolution percentages
# and a few in-between values.  Keeping this list tight makes the fit both fast
# and less prone to picking a harmonic.
SCALES: tuple[float, ...] = tuple(sorted(set(
    [round(float(v), 4) for v in np.arange(1.050, 1.201, 0.005)]
    + [round(float(v), 4) for v in np.arange(1.800, 2.051, 0.005)]
    + [1.222, 1.250, 1.286, 1.300, 1.333, 1.375, 1.400, 1.429, 1.500, 1.600,
       1.625, 1.667, 1.714, 1.750, 2.100, 2.200, 2.250, 2.400, 2.500, 2.667,
       2.800, 3.000, 3.200, 3.500, 4.000]
)))


def _recon_pair(rows: np.ndarray, s: float, phi: float) -> tuple[float, float]:
    """Fit a source grid twice: piecewise-linear and nearest-neighbour.

    Nearest-neighbour upscaling survives JPEG compression much better than
    exact duplicate pixels do, so this second fit is what rescues mobile /
    screenshot-service recompressed images.
    """
    W = rows.shape[1]
    k = int((W - 1 - phi) / s) + 1
    if k < 12:
        return 1.0, 1.0
    xs = phi + np.arange(k, dtype=np.float32) * s
    i0 = np.clip(np.floor(xs).astype(np.int32), 0, W - 2)
    f = xs - i0
    lo = rows[:, i0]
    lival = lo + (rows[:, i0 + 1] - lo) * f
    t = (np.arange(W, dtype=np.float32) + 0.5 - phi) / s
    j = np.floor(t).astype(np.int32)
    w = t - j
    j = np.clip(j, 0, k - 2)
    a = lival[:, j]
    rec_lin = a + (lival[:, j + 1] - a) * w
    rec_near = a

    def trimmed(rec):
        per_row = ((rows - rec) ** 2).mean(axis=1)
        # drop the worst 30% of rows: HUD text and subtitles are outliers
        if per_row.size >= 6:
            keep = max(4, int(per_row.size * 0.7))
            per_row = np.sort(per_row)[:keep]
        return float(per_row.mean())

    return trimmed(rec_lin), trimmed(rec_near)


def _best_pair(rows: np.ndarray, s: float, phases: int) -> tuple[float, float, float, float]:
    best_lin, best_lin_phi = 1e30, 0.0
    best_near, best_near_phi = 1e30, 0.0
    for i in range(phases):
        phi = i * s / phases
        e_lin, e_near = _recon_pair(rows, s, phi)
        if e_lin < best_lin:
            best_lin, best_lin_phi = e_lin, phi
        if e_near < best_near:
            best_near, best_near_phi = e_near, phi
    return best_lin, best_lin_phi, best_near, best_near_phi


def _select_rows(gray: np.ndarray, axis: int, count: int = 40) -> np.ndarray:
    g = gray if axis == 1 else gray.T
    H = g.shape[0]
    idx = np.linspace(0, H - 1, min(count, H)).astype(int)
    rows = g[idx].astype(np.float32)
    # drop featureless rows: they add error without adding grid information
    std = rows.std(axis=1)
    grad = np.abs(np.diff(rows, axis=1)).mean(axis=1)
    keep = (std > 0.02) & (grad > 0.004)
    if keep.sum() >= 8:
        rows = rows[keep]
    return rows


def _recon_axis(gray: np.ndarray, axis: int) -> tuple[AxisEvidence, list[tuple[float, float, float]]]:
    name = "x" if axis == 1 else "y"
    rows = _select_rows(gray, axis)
    if rows.shape[1] < 200 or rows.shape[0] < 6:
        return AxisEvidence(name, "none"), []
    base = float(np.mean(rows ** 2)) or 1.0
    curves: dict[str, list[tuple[float, float, float]]] = {"linear": [], "nearest": []}
    for s in SCALES:
        e_lin, _pl, e_near, _pn = _best_pair(rows, s, 14)
        el_lin, _, el_near, _ = _best_pair(rows, s - 0.10, 6)
        eh_lin, _, eh_near, _ = _best_pair(rows, s + 0.10, 6)
        prom_lin = (min(el_lin, eh_lin) - e_lin) / max(e_lin, 1e-12)
        prom_near = (min(el_near, eh_near) - e_near) / max(e_near, 1e-12)
        curves["linear"].append((s, e_lin / base, prom_lin))
        curves["nearest"].append((s, e_near / base, prom_near))
    # A true grid shows a dip that is significantly below *both* neighbours,
    # and the residual must be small in absolute terms: a native frame can
    # also wiggle, but its reconstruction error stays high.
    ceilings = {"linear": 0.0035, "nearest": 0.0025}
    thresholds = {"linear": 0.10, "nearest": 0.18}
    chosen: dict[str, tuple[float, float, float]] = {}
    for kernel, curve in curves.items():
        good = [c for c in curve if c[2] >= thresholds[kernel] and c[1] <= ceilings[kernel]]
        if good:
            chosen[kernel] = min(good, key=lambda c: c[1])
    if not chosen:
        return AxisEvidence(name, "none"), curves["linear"]
    if "linear" in chosen and "nearest" in chosen:
        lin, near = chosen["linear"], chosen["nearest"]
        if abs(lin[0] - near[0]) <= 0.03:
            # same scale: a smooth fit is the more conservative explanation
            pick, kernel = lin, "linear"
        elif near[2] > 2.0 * lin[2] and near[1] < lin[1]:
            pick, kernel = near, "nearest"
        else:
            pick, kernel = (lin, "linear") if lin[1] <= near[1] else (near, "nearest")
    else:
        kernel = next(iter(chosen))
        pick = chosen[kernel]
    s, e, prom = pick
    near_native = False
    # A 7% upscale is almost indistinguishable from native, and a too-coarse
    # candidate (2x) can win on flat drawings.  If there is a credible local
    # minimum just above 1.0, prefer it over a suspicious 1.1-1.35 or an
    # unclean 2x result.
    all_c = [(ss, ee, pp, kk) for kk, curve in curves.items() for ss, ee, pp in curve]
    gmin = min(c[1] for c in all_c) if all_c else 1e30
    near = [c for c in all_c
            if 1.055 <= c[0] <= 1.105 and c[2] >= 0.12 and c[1] <= 3.0 * gmin]
    if near and ((1.12 <= s <= 1.35) or (s >= 1.9 and prom < 0.5)):
        near.sort(key=lambda c: (c[0], c[1]))
        s, e, prom, kernel = near[0]
        near_native = True
    _, phi_lin, _, phi_near = _best_pair(rows, s, 16)
    method = "nearest" if kernel == "nearest" else "smooth"
    phi = phi_near if kernel == "nearest" else phi_lin
    ev = AxisEvidence(name, method, float(s), prominence=float(prom), phase=float(phi),
                      match=float(min(0.99, 0.75 + prom)), near_native=near_native)
    return ev, curves[kernel]


def _block_variance_2x(gray: np.ndarray) -> float:
    """Fraction of variance left inside 2x2 blocks.

    A screenshot that was upscaled 2x by the OS/compositor (very common on
    high-DPI tablets) has almost no variance inside each 2x2 block.  This
    detector works even when the 2x filter is smooth and the reconstruction
    model in :func:`_recon_axis` cannot see a sharp grid.
    """
    h = (gray.shape[0] // 2) * 2
    w = (gray.shape[1] // 2) * 2
    if h < 16 or w < 16:
        return 1.0
    g = gray[:h, :w]
    blocks = g.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3).reshape(h // 2, w // 2, 4)
    within = float(blocks.var(axis=2).mean())
    total = float(g.var())
    return within / total if total > 1e-12 else 1.0


def _unwrap_2x(rgb: np.ndarray, outer: AnalysisResult, path: str, quick: bool) -> AnalysisResult | None:
    """Undo a 2x screenshot/OS upscale, then analyse the logical framebuffer.

    The game's 3D render may itself be lower than that framebuffer (for
    example 1.5x), so the total scale is 2 * inner.
    """
    h, w = rgb.shape[:2]
    best: tuple[float, AnalysisResult] | None = None
    for oy, ox in ((0, 0), (1, 1)):
        half = np.ascontiguousarray(rgb[oy:h:2, ox:w:2])
        inner = analyse_array(half, path=path, quick=quick, _depth=1)
        score = inner.confidence + (0.25 if inner.method != "none" else 0.0)
        if best is None or score > best[0]:
            best = (score, inner)
    if best is None:
        return None
    inner = best[1]
    combine = inner.method != "none" and abs(inner.scale - 2.0) >= 0.06
    inner_scale = inner.scale if combine else 1.0
    total = 2.0 * inner_scale
    res = inner
    res.width, res.height = w, h
    res.content_box = (0, 0, w, h)
    res.bars = (0, 0, 0, 0)
    res.scale = total
    res.scale_x = 2.0 * (inner.scale_x if combine else 1.0)
    res.scale_y = 2.0 * (inner.scale_y if combine else 1.0)
    res.internal_w = int(round(w / res.scale_x))
    res.internal_h = int(round(h / res.scale_y))
    res.percent = 100.0 / total
    if combine:
        res.method = inner.method
        res.confidence = float(np.clip(inner.confidence * 0.95, 0.4, 0.97))
    else:
        res.method = "nearest" if (outer.method == "nearest" and abs(outer.scale - 2.0) < 0.06) else "smooth"
        res.confidence = max(0.70, outer.confidence)
    # report the inner grid in the coordinates of the full screenshot
    res.recon_x = [(s * 2.0, e, p) for s, e, p in inner.recon_x] if combine else []
    res.recon_y = [(s * 2.0, e, p) for s, e, p in inner.recon_y] if combine else []
    for ev in (res.evidence_x, res.evidence_y):
        if ev.method != "none" and combine:
            ev.scale *= 2.0
    ev_inner = "；内部还有 %.3f× 渲染放大" % inner_scale if combine else ""
    res.notes.insert(0, f"检测到 2× 截图/系统放大{ev_inner}，内部分析在 {w//2}×{h//2} 上进行")
    res.candidates = _resolution_candidates(w, h, res.scale_x, res.scale_y)
    res.elapsed = outer.elapsed + inner.elapsed
    return res

def _detect_bars(rgb: np.ndarray) -> tuple[int, int, int, int]:
    h, w = rgb.shape[:2]
    gray = _luma(rgb)
    lim = 9.0 / 255.0

    def run(means: np.ndarray, stds: np.ndarray, n: int) -> int:
        k = 0
        while k < n - 2 and means[k] < lim and stds[k] < 0.008:
            k += 1
        return k

    mt, st = gray.mean(axis=1), gray.std(axis=1)
    mb, sb = gray[::-1].mean(axis=1), gray[::-1].std(axis=1)
    ml, sl = gray.mean(axis=0), gray.std(axis=0)
    mr, sr = gray[:, ::-1].mean(axis=0), gray[:, ::-1].std(axis=0)
    top, bottom = run(mt, st, h), run(mb, sb, h)
    left, right = run(ml, sl, w), run(mr, sr, w)
    # Only trust symmetric pairs: a single dark strip is usually part of the
    # scene (a dark floor, a photo's vignette), not a letterbox.
    if not (top > 0 and bottom > 0):
        top = bottom = 0
    if not (left > 0 and right > 0):
        left = right = 0
    if top + bottom > h * 0.45:
        top = bottom = 0
    if left + right > w * 0.45:
        left = right = 0
    return top, bottom, left, right


_LADDER_16_9 = [
    (640, 360), (854, 480), (960, 540), (1024, 576), (1152, 648),
    (1280, 720), (1366, 768), (1440, 810), (1600, 900), (1680, 945),
    (1920, 1080), (2133, 1200), (2560, 1440), (2880, 1620),
    (3200, 1800), (3840, 2160), (5120, 2880), (7680, 4320),
]
_LADDER_21_9 = [(2560, 1080), (3440, 1440), (3840, 1600), (5120, 2160)]
_LADDER_16_10 = [(1280, 800), (1440, 900), (1680, 1050), (1920, 1200), (2560, 1600), (3840, 2400)]


def _resolution_candidates(w: int, h: int, sx: float, sy: float) -> list[ResolutionCandidate]:
    iw, ih = w / sx, h / sy
    target_ar = w / h
    ladder = list(_LADDER_16_9)
    if abs(target_ar - 21 / 9) < 0.15:
        ladder += _LADDER_21_9
    if abs(target_ar - 16 / 10) < 0.08:
        ladder += _LADDER_16_10
    out: list[ResolutionCandidate] = []
    seen: set[tuple[int, int]] = set()
    for rw, rh in ladder:
        ar = rw / rh
        if abs(ar - target_ar) / target_ar > 0.05:
            continue
        s = (w / rw + h / rh) / 2
        if s > 9 or s < 1.02:
            continue
        err = abs(np.log(s / ((sx + sy) / 2)))
        if err > 0.09:
            continue
        common = 1.0
        if rh in (720, 1080, 1440, 2160, 900, 540, 810, 1200, 1800, 1620):
            common = 1.2
        if rh in (810, 900, 540, 720, 1080, 1440):
            common += 0.1
        score = float(np.exp(-((err / 0.030) ** 2)) * common)
        if (rw, rh) in seen:
            continue
        seen.add((rw, rh))
        out.append(ResolutionCandidate(rw, rh, s, score, f"{rw}×{rh}"))
    ew, eh = int(round(iw)), int(round(ih))
    if ew > 0 and eh > 0:
        out.append(ResolutionCandidate(ew, eh, (sx + sy) / 2, 0.5, "精确推断", "exact"))
    out.sort(key=lambda c: -c.score)
    unique: list[ResolutionCandidate] = []
    for c in out:
        if any(abs(c.width - u.width) <= 2 and abs(c.height - u.height) <= 2 for u in unique):
            continue
        unique.append(c)
    return unique[:6]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def analyse_array(rgb: np.ndarray, *, path: str = "", quick: bool = False,
                  base_size: tuple[int, int] | None = None,
                  roi: tuple[int, int, int, int] | None = None,
                  _depth: int = 0) -> AnalysisResult:
    t0 = time.perf_counter()
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    rgb = rgb[..., :3]
    H, W = rgb.shape[:2]
    base_w, base_h = base_size or (W, H)
    bars = _detect_bars(rgb)
    top, bottom, left, right = bars
    content = rgb[top:H - bottom, left:W - right]
    cw, ch = content.shape[1], content.shape[0]
    gray = _luma(content)
    notes: list[str] = []
    if any(bars):
        notes.append(f"已裁掉黑边：上 {top} · 下 {bottom} · 左 {left} · 右 {right} px")
    if roi is not None:
        notes.append(f"只分析选区 {roi[2]-roi[0]}×{roi[3]-roi[1]}"
                     f"（{roi[0]},{roi[1]}），再按整图 {base_w}×{base_h} 换算")

    # Ignore the outer 6%: HUD and letterbox remnants are usually native-res.
    mx, my = int(cw * 0.05), int(ch * 0.05)
    work = gray[my:ch - my, mx:cw - mx] if cw > 60 and ch > 60 else gray

    res = AnalysisResult(
        width=base_w, height=base_h,
        content_box=roi if roi is not None else (left, top, left + cw, top + ch),
        bars=bars, path=path, roi=roi,
    )

    evx = _nearest_axis(work, 1)
    evy = _nearest_axis(work, 0)
    rx, cx = _recon_axis(work, 1)
    ry, cy = _recon_axis(work, 0)
    res.recon_x, res.recon_y = cx, cy

    # --- choose a detector / scale ----------------------------------------
    method = "none"
    near_used = False
    scale_x = scale_y = 1.0
    nearest_ok_x = evx.method == "nearest" and evx.support >= 2 and evx.match >= 0.995
    nearest_ok_y = evy.method == "nearest" and evy.support >= 2 and evy.match >= 0.995
    recon_ok_x = ((rx.method == "nearest" and rx.prominence >= 0.18)
                  or (rx.method == "smooth" and rx.prominence >= 0.10))
    recon_ok_y = ((ry.method == "nearest" and ry.prominence >= 0.18)
                  or (ry.method == "smooth" and ry.prominence >= 0.10))

    if nearest_ok_x or nearest_ok_y:
        # Exact duplicate pixels: the strongest possible evidence.
        method = "nearest"
        if nearest_ok_x and nearest_ok_y:
            scale_x, scale_y = evx.scale, evy.scale
        elif nearest_ok_x:
            scale_x = scale_y = evx.scale
        else:
            scale_x = scale_y = evy.scale
        if abs(scale_x - scale_y) / max(scale_x, scale_y) > 0.03:
            notes.append("横纵重复比例不同，可能启用了非等比缩放")
    elif recon_ok_x or recon_ok_y:
        near_used = rx.near_native or ry.near_native
        if near_used:
            win = rx if rx.near_native else ry
            scale_x = scale_y = win.scale
            method = win.method
            notes.append("检测到接近原生的放大倍率（<20%），这类截图的可分辨余量很小，置信度有限")
        elif recon_ok_x and recon_ok_y:
            if abs(rx.scale - ry.scale) / max(rx.scale, ry.scale) > 0.03:
                # Prefer the smaller scale when both axes have real evidence:
                # a 2x fit is often a harmonic of a true 1.8-1.95x grid.
                win = rx if rx.prominence >= ry.prominence else ry
                if min(rx.prominence, ry.prominence) >= 0.4 * max(rx.prominence, ry.prominence):
                    win = rx if rx.scale < ry.scale else ry
                scale_x = scale_y = win.scale
                method = win.method
                notes.append("横纵拟合结果略有差异，取证据更强的一侧")
            else:
                scale_x = scale_y = (rx.scale + ry.scale) / 2
                method = "nearest" if "nearest" in (rx.method, ry.method) else "smooth"
        elif recon_ok_x:
            scale_x = scale_y = rx.scale
            method = rx.method
        else:
            scale_x = scale_y = ry.scale
            method = ry.method
    else:
        res.native_likely = True
        notes.append("没有找到稳定的周期性像素网格：可能是原生渲染，或使用了 DLSS/TSR 等时域重建")
        # a very weak duplicate hint is still worth showing
        if evx.method == "nearest" and evx.scale > 1.05 and evx.support >= 2:
            notes.append(f"弱提示：约 {evx.scale:.3f}×，但证据不足")

    if method != "none":
        res.method = method
        res.scale_x = float(np.clip(scale_x, 1.0, 10.0))
        res.scale_y = float(np.clip(scale_y, 1.0, 10.0))
        res.scale = float(np.clip((res.scale_x + res.scale_y) / 2, 1.0, 10.0))
        res.internal_w = int(round(base_w / res.scale_x))
        res.internal_h = int(round(base_h / res.scale_y))
        res.percent = 100.0 / res.scale
        if method == "nearest":
            if nearest_ok_x or nearest_ok_y:
                res.evidence_x = evx if evx.method != "none" else AxisEvidence("x", "nearest", res.scale_x)
                res.evidence_y = evy if evy.method != "none" else AxisEvidence("y", "nearest", res.scale_y)
                support = min(evx.support, evy.support) if (nearest_ok_x and nearest_ok_y) else max(evx.support, evy.support)
                res.confidence = float(np.clip(0.88 + 0.02 * min(support, 5), 0.0, 0.97))
                if support < 2:
                    res.confidence = 0.72
            else:
                res.evidence_x = rx if rx.method != "none" else AxisEvidence("x", "nearest", res.scale_x)
                res.evidence_y = ry if ry.method != "none" else AxisEvidence("y", "nearest", res.scale_y)
                prom = max(rx.prominence if rx.method != "none" else 0.0,
                           ry.prominence if ry.method != "none" else 0.0)
                res.confidence = float(np.clip(0.78 + 0.18 * min(prom, 1.0), 0.45, 0.96))
                res.notes.append(
                    f"像素级放大（最近邻 / 整数缩放）；重建误差在 {res.scale:.3f}× 处触底，"
                    "说明画面被整块复制放大")
            if near_used:
                res.confidence = min(res.confidence, 0.55)
        else:
            res.evidence_x = rx if rx.method != "none" else AxisEvidence("x", "smooth", res.scale_x)
            res.evidence_y = ry if ry.method != "none" else AxisEvidence("y", "smooth", res.scale_y)
            res.evidence_x.method = res.evidence_y.method = "smooth"
            res.evidence_x.scale, res.evidence_y.scale = res.scale_x, res.scale_y
            prom = max(rx.prominence, ry.prominence)
            res.confidence = float(np.clip(0.58 + 0.55 * min(prom, 0.7), 0.35, 0.88))
            res.notes.append("平滑放大（双线性 / 双三次 / FSR 一类），网格比像素级放大更弱，结果可能有 ±1 档误差")
            if near_used:
                res.confidence = min(res.confidence, 0.55)
        if res.percent < 20 or res.scale > 4.5:
            res.confidence *= 0.85
            notes.append("缩放倍数很大，请优先用证据图核对")
        if res.scale < 1.20:
            res.confidence = min(res.confidence, 0.55)
            notes.append("倍率接近 1（<20%）：这类缩放与原生很难区分，建议用无损原图或更清晰的画面复核")
    else:
        res.scale = 1.0
        res.internal_w, res.internal_h = base_w, base_h
        res.percent = 100.0
        res.confidence = 0.55
        res.evidence_x = evx if evx.method != "none" else rx
        res.evidence_y = evy if evy.method != "none" else ry

    res.candidates = _resolution_candidates(base_w, base_h, res.scale_x, res.scale_y)
    if method != "none" and res.candidates:
        best = res.candidates[0]
        if abs(best.width - res.internal_w) > 3 or abs(best.height - res.internal_h) > 3:
            notes.append(f"最接近的常见档位：{best.label}")

    # --- evidence: duplicate gaps / folded pattern -------------------------
    d1x, _ = _profiles(gray, 1)
    if method == "nearest" and (nearest_ok_x or nearest_ok_y):
        res.duplicate_gaps = _duplicate_gaps(d1x)
        res.pattern = _fold_pattern(d1x, res.scale_x)
        res.evidence_x.period = res.evidence_x.period or 0
    elif method in ("smooth", "nearest"):
        res.pattern = _fold_pattern(_profiles(gray, 1)[1], res.scale_x)

    # --- dynamic-resolution tile map (exact duplicate evidence only) -------
    if not quick and method == "nearest" and (nearest_ok_x or nearest_ok_y):
        ts = max(96, min(256, min(ch, cw) // 4))
        gx, gy = max(1, cw // ts), max(1, ch // ts)
        tile_map = [[0.0] * gx for _ in range(gy)]
        for j in range(gy):
            for i in range(gx):
                t = gray[j * ts:(j + 1) * ts, i * ts:(i + 1) * ts]
                if t.std() < 0.035:
                    continue
                cand = _dup_search(_profiles(t, 1)[0], max_period=12)
                if cand:
                    tile_map[j][i] = cand[0].scale
        res.tile_map = tile_map
        res.tile_grid = (gx, gy)

    res.notes = notes + res.notes
    if method != "none":
        if roi is not None:
            res.notes.insert(0, f"截图 {base_w}×{base_h}，只分析选区 "
                                f"{roi[2]-roi[0]}×{roi[3]-roi[1]}")
        else:
            res.notes.insert(0, f"截图 {W}×{H}，分析区域 {cw}×{ch}")
    if _depth == 0 and roi is None and base_size is None:
        bv = _block_variance_2x(gray)
        if bv < 0.045 and (res.method == "none" or
                           (res.method == "nearest" and abs(res.scale - 2.0) < 0.06)):
            combined = _unwrap_2x(content, res, path, quick)
            if combined is not None:
                combined.elapsed = time.perf_counter() - t0
                return combined
    res.elapsed = time.perf_counter() - t0
    return res


def _fold_pattern(profile: np.ndarray, scale: float) -> list[float]:
    if scale < 1.01 or profile.size < 8:
        return []
    base = _moving_average(profile, 21)
    r = profile / np.maximum(base, 1e-9)
    if abs(scale - round(scale)) > 0.02:
        period = max(2, int(round(scale * 4)))
    else:
        period = int(round(scale))
    period = int(np.clip(period, 2, 24))
    idx = np.arange(r.size) % period
    cnt = np.bincount(idx, minlength=period)
    m = np.bincount(idx, weights=r, minlength=period) / np.maximum(cnt, 1)
    m = m / max(1e-9, m.mean())
    return [float(v) for v in m]


def _duplicate_gaps(d1: np.ndarray) -> list[int]:
    """Distances between source-pixel boundaries (stair tread widths).

    For a 1.5x nearest-neighbour upscale the values alternate 1, 2, 1, 2 ...:
    the histogram peaks at 1 and 2 px and its mean is the scale factor.
    """
    thr = max(1e-6, float(np.percentile(d1, 90)) * 0.02)
    pos = np.flatnonzero(d1 > thr)
    if pos.size < 4:
        return []
    gaps = np.diff(pos)
    gaps = gaps[(gaps >= 1) & (gaps <= 12)]
    if gaps.size > 4000:
        gaps = np.random.default_rng(0).choice(gaps, 4000, replace=False)
    return [int(v) for v in gaps]


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_image(path: str) -> np.ndarray:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (0, 0, 0))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")
    return np.asarray(im)


def analyse_image(path: str, *, quick: bool = False) -> AnalysisResult:
    return analyse_array(load_image(path), path=path, quick=quick)


def analyse_region(rgb: np.ndarray, roi: tuple[int, int, int, int], *,
                   path: str = "", quick: bool = False) -> AnalysisResult:
    """Analyse only ``roi`` but report resolution against the full image.

    ``roi`` is ``(x0, y0, x1, y1)`` in full-image coordinates.
    """
    arr = np.asarray(rgb)
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in roi)
    x0, y0 = max(0, min(x0, w - 2)), max(0, min(y0, h - 2))
    x1, y1 = max(x0 + 2, min(x1, w)), max(y0 + 2, min(y1, h))
    crop = np.ascontiguousarray(arr[y0:y1, x0:x1])
    return analyse_array(crop, path=path, quick=quick,
                         base_size=(w, h), roi=(x0, y0, x1, y1))

