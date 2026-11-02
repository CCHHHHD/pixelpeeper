"""Rendering of the shareable evidence card and small in-app charts."""
from __future__ import annotations

import math
import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .analysis import AnalysisResult

BG = (11, 15, 20)
PANEL = (20, 26, 33)
LINE = (42, 52, 63)
TEXT = (232, 238, 244)
MUTED = (137, 150, 163)
ACCENT = (72, 214, 171)
ACCENT2 = (240, 178, 90)
RED = (238, 110, 110)

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
_BOLD_CANDIDATES = [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc"]
_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key in _cache:
        return _cache[key]
    paths = _BOLD_CANDIDATES if bold else _FONT_CANDIDATES
    for p in paths:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _cache[key] = f
                return f
            except Exception:
                pass
    f = ImageFont.load_default()
    _cache[key] = f
    return f


def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_w: int, size: int) -> str:
    f = font(size)
    if draw.textlength(text, font=f) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=f) > max_w:
        text = text[:-1]
    return text + "…"


def _confidence_color(c: float) -> tuple[int, int, int]:
    if c >= 0.8:
        return ACCENT
    if c >= 0.55:
        return ACCENT2
    return RED


def _panel(draw, box, fill=PANEL, radius=16):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=LINE, width=1)


def _stat(draw, x, y, label, value, sub="", w=200):
    draw.text((x, y), label, font=font(15), fill=MUTED)
    draw.text((x, y + 22), value, font=font(28, True), fill=TEXT)
    if sub:
        draw.text((x, y + 58), sub, font=font(13), fill=MUTED)


def _grid_overlay(im: Image.Image, scale: float, phase: float,
                  origin: tuple[float, float] = (0.0, 0.0),
                  phase_y: float | None = None) -> Image.Image:
    """Draw faint lines at the inferred source-pixel boundaries."""
    if scale < 1.05:
        return im
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    w, h = im.size
    x = (phase - origin[0]) % scale
    while x < w:
        d.line([(x, 0), (x, h)], fill=(72, 214, 171, 72), width=1)
        x += scale
    py = phase if phase_y is None else phase_y
    y = (py - origin[1]) % scale
    while y < h:
        d.line([(0, y), (w, y)], fill=(72, 214, 171, 45), width=1)
        y += scale
    return Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")


def _best_crop(pil: Image.Image, size: int) -> tuple[int, int, int, int]:
    """Pick the most contrasty square: a good place to see the pixel grid."""
    g = pil.convert("L").filter(ImageFilter.GaussianBlur(1.2))
    w, h = g.size
    s = min(size, w, h)
    if w <= s or h <= s:
        return (0, 0, w, h)
    small = g.resize((max(2, w // 8), max(2, h // 8)))
    arr = np.asarray(small, np.float32)
    ii = arr.cumsum(0).cumsum(1)
    ii2 = (arr * arr).cumsum(0).cumsum(1)
    k = max(2, s // 8)

    def box(ii_, y, x):
        y2, x2 = min(y + k, arr.shape[0] - 1), min(x + k, arr.shape[1] - 1)
        total = ii_[y2, x2]
        if y:
            total -= ii_[y - 1, x2]
        if x:
            total -= ii_[y2, x - 1]
        if y and x:
            total += ii_[y - 1, x - 1]
        return total

    best = (0, 0, -1.0)
    for y in range(0, arr.shape[0] - k, max(1, k // 2)):
        for x in range(0, arr.shape[1] - k, max(1, k // 2)):
            area = k * k
            mean = box(ii, y, x) / area
            var = box(ii2, y, x) / area - mean * mean
            # prefer a bright, contrasty patch: dark corners hide the grid
            score = var * (0.05 + (mean / 255.0) ** 2)
            if score > best[2]:
                best = (x, y, score)
    x = int(best[0] * 8 - s / 2)
    y = int(best[1] * 8 - s / 2)
    x = max(0, min(w - s, x))
    y = max(0, min(h - s, y))
    return (x, y, x + s, y + s)


def build_card(result: AnalysisResult, pil: Image.Image, filename: str = "") -> Image.Image:
    W, H = 1600, 980
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 4], fill=ACCENT)
    d.text((44, 30), "数毛器", font=font(34, True), fill=TEXT)
    d.text((170, 38), "PIXELPEEPER · 截图像素网格分析", font=font(16), fill=MUTED)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    d.text((W - 44 - d.textlength(stamp, font=font(14)), 42), stamp, font=font(14), fill=MUTED)
    name = _fit_text(d, filename or "screenshot", 760, 16)
    d.text((44, 82), name, font=font(16), fill=MUTED)

    _panel(d, (40, 116, W - 40, 330), PANEL)
    if result.method == "none":
        title = "未发现放大网格"
        sub = f"截图 {result.width} × {result.height} · 可能是原生渲染，或使用了 DLSS / TSR 等时域重建"
        big = f"{result.width} × {result.height}"
    else:
        title = "推断内部分辨率"
        big = f"{result.internal_w} × {result.internal_h}"
        kind = "像素级放大" if result.method == "nearest" else "平滑放大"
        sub = f"{result.percent:.1f}%  ·  {result.scale:.3f}×  ·  {kind}"
    d.text((72, 146), title, font=font(17), fill=MUTED)
    d.text((70, 172), big, font=font(64, True), fill=TEXT)
    d.text((72, 262), sub, font=font(18), fill=ACCENT if result.method != "none" else MUTED)

    cx, cy, r = W - 155, 222, 62
    d.arc([cx - r, cy - r, cx + r, cy + r], 0, 360, fill=(45, 55, 66), width=10)
    d.arc([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * result.confidence,
          fill=_confidence_color(result.confidence), width=10)
    conf = f"{round(result.confidence * 100)}%"
    d.text((cx - d.textlength(conf, font=font(30, True)) / 2, cy - 26), conf,
           font=font(30, True), fill=TEXT)
    d.text((cx - d.textlength("置信度", font=font(13)) / 2, cy + 14), "置信度",
           font=font(13), fill=MUTED)

    sx = 830
    _stat(d, sx, 150, "截图分辨率", f"{result.width} × {result.height}",
          f"内容区 {result.content_box[2]-result.content_box[0]} × {result.content_box[3]-result.content_box[1]}")
    _stat(d, sx + 205, 150, "横向倍数", f"{result.scale_x:.3f}×", "源像素 / 屏幕像素")
    _stat(d, sx + 390, 150, "纵向倍数", f"{result.scale_y:.3f}×", "通常与横向一致")
    _stat(d, sx + 205, 240, "等效渲染比例", f"{result.percent:.1f}%", "100% = 原生")
    _stat(d, sx + 390, 240, "分析耗时", f"{result.elapsed:.1f}s", "本地 CPU 计算")

    _panel(d, (40, 354, 700, 860), PANEL)
    d.text((64, 374), "局部放大", font=font(18, True), fill=TEXT)
    d.text((136, 377), "青线 = 推断的源像素边界" if result.method == "nearest"
           else "平滑插值后的源像素边界", font=font(13), fill=MUTED)
    side = 133
    if result.roi:
        roi_img = pil.crop(result.roi)
        local = _best_crop(roi_img, min(side, roi_img.width, roi_img.height))
        box = (result.roi[0] + local[0], result.roi[1] + local[1],
               result.roi[0] + local[2], result.roi[1] + local[3])
    else:
        box = _best_crop(pil, side)
    crop = pil.crop(box)
    zoom = max(2, int(400 // max(1, min(crop.size))))
    z = crop.resize((crop.width * zoom, crop.height * zoom), Image.NEAREST)
    ox, oy = 64, 412
    if result.method == "nearest" and result.scale >= 1.05:
        rx0 = result.roi[0] if result.roi else 0
        ry0 = result.roi[1] if result.roi else 0
        z = _grid_overlay(z, result.scale * zoom,
                          (rx0 + (result.evidence_x.phase or 0.0)) * zoom,
                          (box[0] * zoom, box[1] * zoom),
                          (ry0 + (result.evidence_y.phase or 0.0)) * zoom)
    shown = z.crop((0, 0, min(z.width, 420), min(z.height, 384)))
    im.paste(shown, (ox, oy))
    d.rectangle([ox - 1, oy - 1, ox + shown.width, oy + shown.height], outline=LINE)
    d.text((64, 812), f"{zoom}× 最近邻放大 · 截取自 ({box[0]}, {box[1]})", font=font(13), fill=MUTED)
    if result.method == "nearest":
        d.text((64, 834), "斜边台阶的宽度 = 源像素的宽度", font=font(13), fill=MUTED)
    else:
        d.text((64, 834), "平滑放大的台阶更细，靠右侧误差曲线确认", font=font(13), fill=MUTED)

    _panel(d, (720, 354, W - 40, 700), PANEL)
    if result.method == "nearest" and result.duplicate_gaps:
        d.text((744, 374), "台阶宽度分布", font=font(18, True), fill=TEXT)
        d.text((896, 377), f"平均应落在 {result.scale:.2f} px 附近", font=font(13), fill=MUTED)
        _draw_gap_chart(d, (760, 420, W - 80, 660), result)
    elif result.method in ("smooth", "nearest") and (result.recon_x or result.recon_y):
        d.text((744, 374), "重采样拟合误差", font=font(18, True), fill=TEXT)
        d.text((928, 377), "误差在真实网格处触底，越低越可信", font=font(13), fill=MUTED)
        _draw_recon_chart(d, (770, 420, W - 70, 665), result)
    else:
        d.text((744, 374), "没有稳定的周期网格", font=font(18, True), fill=TEXT)
        for i, line in enumerate([
            "原生渲染的像素之间没有固定的重复关系；",
            "DLSS / TSR / FSR2 等时域重建也会打散网格。",
            "可以换一张主体清晰、HUD 较少的截图再试。",
        ]):
            d.text((744, 430 + i * 30), line, font=font(16), fill=MUTED)

    _panel(d, (720, 714, W - 40, 860), PANEL)
    d.text((744, 730), "常见档位", font=font(16, True), fill=TEXT)
    x = 744
    for c in result.candidates[:4]:
        label = f"{c.width}×{c.height}"
        col = ACCENT if c.score > 0.8 else (TEXT if c.score > 0.45 else MUTED)
        d.text((x, 760), label, font=font(17, True), fill=col)
        d.text((x, 786), f"{100 / c.scale:.1f}%", font=font(13), fill=MUTED)
        x += 190
    if result.tile_map:
        gx, gy = result.tile_grid
        mx0, my0, cell = W - 245, 728, 14
        for j in range(gy):
            for i in range(gx):
                v = result.tile_map[j][i]
                if v <= 0:
                    col = (30, 38, 47)
                else:
                    t = min(1.0, max(0.0, (v - 1.0) / 2.0))
                    col = (int(40 + 40 * t), int(80 + 120 * t), int(90 + 70 * t))
                d.rectangle([mx0 + i * cell, my0 + j * cell,
                             mx0 + (i + 1) * cell - 2, my0 + (j + 1) * cell - 2], fill=col)
        d.text((mx0, my0 + gy * cell + 4), "分块一致性", font=font(11), fill=MUTED)

    y = 882
    for note in result.notes[:2]:
        d.text((44, y), "· " + _fit_text(d, note, 1480, 14), font=font(14), fill=MUTED)
        y += 24
    d.text((W - 44 - d.textlength("数毛器 PixelPeeper", font=font(13)), H - 24),
           "数毛器 PixelPeeper", font=font(13), fill=(80, 92, 104))
    return im


def _draw_gap_chart(d, box, result: AnalysisResult):
    x0, y0, x1, y1 = box
    maxg = min(8, max(4, int(math.ceil(result.scale * 2))))
    hist = np.zeros(maxg + 1)
    for g in result.duplicate_gaps:
        if 1 <= g <= maxg:
            hist[g] += 1
    top = max(1.0, hist.max())
    for i in range(1, 4):
        yy = y1 - (y1 - y0) * i / 4
        d.line([(x0, yy), (x1, yy)], fill=(33, 41, 50))
    bw = (x1 - x0) / (maxg + 0.5)
    for g in range(1, maxg + 1):
        h = (y1 - y0) * hist[g] / top
        bx = x0 + (g - 1) * bw + bw * 0.2
        col = ACCENT if abs(g - result.scale) < 0.25 else (54, 148, 126)
        d.rectangle([bx, y1 - h, bx + bw * 0.6, y1], fill=col)
        d.text((bx + bw * 0.3 - d.textlength(str(g), font=font(12)) / 2, y1 + 8),
               str(g), font=font(12), fill=MUTED)
    d.text((x1 - d.textlength("像素 / px", font=font(12)), y1 + 30), "像素 / px",
           font=font(12), fill=MUTED)
    ex = x0 + (result.scale - 1) * bw + bw * 0.5
    d.line([(ex, y0), (ex, y1)], fill=ACCENT2, width=2)
    d.text((ex + 6, y0 + 4), f"{result.scale:.2f}×", font=font(13, True), fill=ACCENT2)


def _draw_recon_chart(d, box, result: AnalysisResult):
    x0, y0, x1, y1 = box

    def series(curve):
        return [(s, max(1e-9, e)) for s, e, _prom in curve if 1.05 <= s <= 4.05]

    sx, sy = series(result.recon_x), series(result.recon_y)
    allv = [e for _, e in sx + sy]
    if not allv:
        return
    lo, hi = math.log10(min(allv)), math.log10(max(allv))
    pad = max(0.08, (hi - lo) * 0.12)
    lo -= pad
    hi += pad

    def X(s): return x0 + (s - 1.0) / 3.1 * (x1 - x0)
    def Y(e): return y1 - (math.log10(max(e, 1e-9)) - lo) / (hi - lo) * (y1 - y0)

    for i in range(1, 5):
        yy = y0 + (y1 - y0) * i / 5
        d.line([(x0, yy), (x1, yy)], fill=(33, 41, 50))
    for s in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        xx = X(s)
        d.line([(xx, y0), (xx, y1)], fill=(30, 37, 45))
        d.text((xx - d.textlength(f"{s:g}×", font=font(11)) / 2, y1 + 8), f"{s:g}×",
               font=font(11), fill=MUTED)
    for pts, col in ((sx, ACCENT), (sy, (118, 160, 235))):
        if len(pts) < 2:
            continue
        xy = [(X(s), Y(e)) for s, e in pts]
        d.line(xy, fill=col, width=2)
        for px, py in xy:
            d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=col)
    if result.scale >= 1.05:
        xx = X(result.scale)
        d.line([(xx, y0 - 2), (xx, y1 + 2)], fill=ACCENT2, width=2)
        d.text((xx + 6, y0 + 2), f"{result.scale:.2f}×", font=font(13, True), fill=ACCENT2)
    d.text((x0, y0 - 18), "相对重建误差（对数）", font=font(11), fill=MUTED)
    d.line([(x1 - 165, y1 + 26), (x1 - 150, y1 + 26)], fill=ACCENT, width=2)
    d.text((x1 - 144, y1 + 19), "横向", font=font(11), fill=MUTED)
    d.line([(x1 - 90, y1 + 26), (x1 - 75, y1 + 26)], fill=(118, 160, 235), width=2)
    d.text((x1 - 69, y1 + 19), "纵向", font=font(11), fill=MUTED)


def mini_chart(result: AnalysisResult, w: int = 420, h: int = 120) -> Image.Image:
    im = Image.new("RGB", (w, h), PANEL)
    d = ImageDraw.Draw(im)
    if result.method == "nearest" and result.duplicate_gaps:
        _draw_small_gap(d, im, result)
    elif result.method in ("smooth", "nearest") and (result.recon_x or result.recon_y):
        _draw_small_recon(d, im, result)
    else:
        d.text((16, 18), "未发现周期性像素网格", font=font(16), fill=MUTED)
        d.text((16, 46), "原生渲染 / 时域重建（DLSS、TSR）", font=font(13), fill=MUTED)
    return im


def _draw_small_gap(d, im, result):
    w, h = im.size
    maxg = min(8, max(4, int(math.ceil(result.scale * 2))))
    hist = np.zeros(maxg + 1)
    for g in result.duplicate_gaps:
        if 1 <= g <= maxg:
            hist[g] += 1
    top = max(1.0, hist.max())
    d.text((14, 10), "台阶宽度分布", font=font(16, True), fill=TEXT)
    bw = (w - 28) / maxg
    for g in range(1, maxg + 1):
        bh = (h - 54) * hist[g] / top
        col = ACCENT if abs(g - result.scale) < 0.25 else (54, 120, 104)
        d.rectangle([14 + (g - 1) * bw + 2, h - 18 - bh, 14 + g * bw - 4, h - 18], fill=col)
        d.text((14 + (g - 1) * bw + bw / 2 - 4, h - 15), str(g), font=font(11), fill=MUTED)
    ex = 14 + (result.scale - 1) * bw + bw / 2
    d.line([(ex, 34), (ex, h - 20)], fill=ACCENT2, width=1)


def _draw_small_recon(d, im, result):
    w, h = im.size
    pts = [(s, max(1e-9, e)) for s, e, _p in result.recon_x if 1.05 <= s <= 4.0]
    if not pts:
        pts = [(s, max(1e-9, e)) for s, e, _p in result.recon_y if 1.05 <= s <= 4.0]
    if not pts:
        return
    lo = math.log10(min(e for _, e in pts))
    hi = math.log10(max(e for _, e in pts))
    if hi - lo < 0.05:
        hi = lo + 0.05
    d.text((14, 10), "拟合误差曲线", font=font(16, True), fill=TEXT)
    xy = []
    for s, e in pts:
        x = 14 + (s - 1.0) / 3.1 * (w - 28)
        y = h - 20 - (math.log10(e) - lo) / (hi - lo) * (h - 52)
        xy.append((x, y))
    d.line(xy, fill=ACCENT, width=2)
    x = 14 + (result.scale - 1.0) / 3.1 * (w - 28)
    d.line([(x, 32), (x, h - 20)], fill=ACCENT2, width=1)


def build_icon() -> Image.Image:
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([2, 2, 62, 62], radius=14, fill=(15, 21, 27, 255))
    for i in range(4):
        for j in range(4):
            col = ACCENT if (i + j) % 3 else (45, 70, 66)
            d.rectangle([10 + i * 11, 10 + j * 11, 18 + i * 11, 18 + j * 11], fill=col)
    d.ellipse([32, 32, 58, 58], outline=ACCENT2, width=4)
    d.line([(50, 50), (61, 61)], fill=ACCENT2, width=4)
    return im
