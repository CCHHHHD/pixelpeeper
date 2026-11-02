"""Self-contained regression tests for the analysis engine.

Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixelpeeper.analysis import analyse_array, analyse_region  # noqa: E402


def make_content(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, h // 8), max(2, w // 8))).astype(np.float32)
    base = np.asarray(
        Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
        np.float32,
    ) / 255.0
    im = Image.fromarray((base * 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    for _ in range(30):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = x0 + rng.integers(w // 20, w // 3), y0 + rng.integers(h // 20, h // 3)
        col = tuple(int(c) for c in rng.integers(20, 240, 3)) + (rng.integers(120, 230),)
        d.rectangle([int(x0), int(y0), int(min(x1, w - 1)), int(min(y1, h - 1))], fill=col)
    for _ in range(20):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        d.line([int(x0), int(y0), int(x1), int(y1)],
               fill=tuple(int(c) for c in rng.integers(160, 255, 3)) + (220,), width=2)
    im = im.filter(ImageFilter.GaussianBlur(0.2))
    arr = np.asarray(im).astype(np.float32)
    arr += np.random.default_rng(seed + 1).normal(0, 16.0, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def nearest(img: np.ndarray, W: int, H: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    xs = np.minimum(((np.arange(W) + 0.5) * iw / W).astype(int), iw - 1)
    ys = np.minimum(((np.arange(H) + 0.5) * ih / H).astype(int), ih - 1)
    return img[ys][:, xs]


def bilinear(img: np.ndarray, W: int, H: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    xs = (np.arange(W, dtype=np.float32) + 0.5) * iw / W - 0.5
    ys = (np.arange(H, dtype=np.float32) + 0.5) * ih / H - 0.5
    x0 = np.clip(np.floor(xs).astype(int), 0, iw - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, ih - 1)
    fx = np.clip(xs - x0, 0, 1)[None, :, None]
    fy = np.clip(ys - y0, 0, 1)[:, None, None]
    a = img[y0][:, x0].astype(np.float32)
    b = img[y0][:, np.minimum(x0 + 1, iw - 1)].astype(np.float32)
    c = img[np.minimum(y0 + 1, ih - 1)][:, x0].astype(np.float32)
    d = img[np.minimum(y0 + 1, ih - 1)][:, np.minimum(x0 + 1, iw - 1)].astype(np.float32)
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    return np.clip(top + (bot - top) * fy, 0, 255).astype(np.uint8)


def make_paint(w: int = 560, h: int = 461) -> Image.Image:
    """A flat, Paint-like drawing: hard edges, text, large uniform areas."""
    im = Image.new("RGB", (w, h), (245, 245, 240))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 30, 220, 140], fill=(220, 60, 60), outline=(20, 20, 20), width=3)
    d.ellipse([260, 40, 420, 180], fill=(60, 120, 220), outline=(10, 10, 10), width=3)
    d.line([30, 300, 520, 220], fill=(20, 160, 80), width=5)
    d.line([50, 430, 500, 250], fill=(240, 160, 20), width=3)
    for i in range(12):
        d.line([30 + i * 42, 350, 30 + i * 42, 440], fill=(30, 30, 30), width=2)
    try:
        f = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 36)
    except Exception:
        f = ImageFont.load_default()
    d.text((60, 180), "数毛测试 560x461", font=f, fill=(10, 10, 10))
    return im


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = make_content(720, 1280, seed=42)

    def test_nearest_2x(self):
        low = make_content(360, 640, seed=1)
        shot = nearest(low, 1280, 720)
        res = analyse_array(shot, quick=True)
        self.assertEqual(res.method, "nearest")
        self.assertAlmostEqual(res.scale, 2.0, delta=0.02)
        self.assertEqual((res.internal_w, res.internal_h), (640, 360))

    def test_nearest_fractional_1_5(self):
        shot = nearest(self.content, 1920, 1080)
        res = analyse_array(shot, quick=True)
        self.assertEqual(res.method, "nearest")
        self.assertAlmostEqual(res.scale, 1.5, delta=0.01)
        self.assertEqual((res.internal_w, res.internal_h), (1280, 720))
        self.assertGreaterEqual(min(res.evidence_x.support, res.evidence_y.support), 2)

    def test_nearest_2x_jpeg(self):
        """A JPEG re-encode destroys exact duplicates; the nearest fit must survive."""
        from io import BytesIO

        low = make_content(360, 640, seed=2)
        shot = nearest(low, 1280, 720)
        buf = BytesIO()
        Image.fromarray(shot).save(buf, "JPEG", quality=85, subsampling=2)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("RGB"))
        res = analyse_array(arr, quick=True)
        # JPEG makes the nearest-vs-bilinear label ambiguous, but the 2x grid
        # itself must survive: this is the mobile-screenshot scenario.
        self.assertIn(res.method, ("nearest", "smooth"))
        self.assertAlmostEqual(res.scale, 2.0, delta=0.02)
        self.assertEqual((res.internal_w, res.internal_h), (640, 360))
        self.assertGreaterEqual(res.confidence, 0.7)

    def test_region_uses_full_image_size(self):
        """A cropped ROI must still report resolution against the full image."""
        low = make_content(360, 640, seed=1)
        full = nearest(low, 1280, 720)
        res = analyse_region(full, (200, 100, 1000, 600), quick=True)
        self.assertEqual(res.method, "nearest")
        self.assertAlmostEqual(res.scale, 2.0, delta=0.02)
        self.assertEqual((res.width, res.height), (1280, 720))
        self.assertEqual((res.internal_w, res.internal_h), (640, 360))
        self.assertEqual(res.roi, (200, 100, 1000, 600))

    def test_near_native_1_07(self):
        """A 7% upscale must not be misread as 2x (the reported regression)."""
        base = make_paint(560, 461)
        shot = np.asarray(base.resize((600, 494), Image.BILINEAR))
        res = analyse_array(shot, quick=True)
        self.assertAlmostEqual(res.scale, 600 / 560, delta=0.012)
        self.assertLessEqual(abs(res.internal_w - 560), 4)
        self.assertLessEqual(abs(res.internal_h - 461), 4)
        self.assertLessEqual(res.confidence, 0.7)

    def test_bilinear_1_5(self):
        shot = bilinear(self.content, 1920, 1080)
        res = analyse_array(shot, quick=True)
        self.assertEqual(res.method, "smooth")
        self.assertAlmostEqual(res.scale, 1.5, delta=0.01)
        self.assertEqual((res.internal_w, res.internal_h), (1280, 720))

    def test_native_is_not_invented(self):
        shot = make_content(1080, 1920, seed=7)
        res = analyse_array(shot, quick=True)
        self.assertEqual(res.method, "none")
        self.assertLess(res.scale, 1.05)


if __name__ == "__main__":
    unittest.main()
