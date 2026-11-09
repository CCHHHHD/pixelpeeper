"""Cross-check the native engine against the Python reference.

This test is development-only: it needs numpy, Pillow and a built pp_cli.
Run from the repository root:

    python native/tests/test_parity.py

Set PP_CLI to the executable if it is not native/build/pp_cli.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pixelpeeper.analysis import analyse_array  # noqa: E402

CLI = os.environ.get(
    "PP_CLI",
    os.path.join(ROOT, "native", "build", "pp_cli" + (".exe" if os.name == "nt" else "")),
)


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


def run_native(arr: np.ndarray, quick: bool, roi=None):
    with tempfile.NamedTemporaryFile(suffix=".rgb", delete=False) as f:
        path = f.name
        arr.tofile(f)
    try:
        cmd = [CLI, path, str(arr.shape[1]), str(arr.shape[0])]
        if quick:
            cmd.append("--quick")
        if roi:
            cmd += ["--roi", ",".join(map(str, roi))]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(proc.stdout)
    finally:
        os.unlink(path)


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CLI):
            raise unittest.SkipTest(f"pp_cli not built: {CLI}")
        cls.content = make_content(720, 1280, seed=42)

    def check(self, arr, *, roi=None, method=None, scale=None, tol=0.025):
        py = analyse_array(arr if roi is None else arr[roi[1]:roi[3], roi[0]:roi[2]],
                           quick=True, base_size=(arr.shape[1], arr.shape[0]) if roi else None,
                           roi=roi if roi else None)
        nat = run_native(arr, True, roi)
        print(f"  python={py.method} {py.scale:.4f} {py.internal_w}x{py.internal_h}; "
              f"native={nat['method']} {nat['scale']:.4f} {nat['internal_w']}x{nat['internal_h']}")
        if method is not None:
            self.assertEqual(nat["method"], method)
        else:
            self.assertEqual(nat["method"], py.method)
        if scale is not None:
            self.assertAlmostEqual(nat["scale"], scale, delta=tol)
        else:
            self.assertAlmostEqual(nat["scale"], py.scale, delta=tol)
        self.assertEqual(nat["internal_w"], py.internal_w)
        self.assertEqual(nat["internal_h"], py.internal_h)
        self.assertAlmostEqual(nat["confidence"], py.confidence, delta=0.10)
        return py, nat

    def test_nearest_2x(self):
        low = make_content(360, 640, seed=1)
        self.check(nearest(low, 1280, 720), method="nearest", scale=2.0)

    def test_nearest_fractional_1_5(self):
        self.check(nearest(self.content, 1920, 1080), method="nearest", scale=1.5)

    def test_nearest_2x_jpeg(self):
        from io import BytesIO
        low = make_content(360, 640, seed=2)
        shot = nearest(low, 1280, 720)
        buf = BytesIO()
        Image.fromarray(shot).save(buf, "JPEG", quality=85, subsampling=2)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("RGB"))
        self.check(arr, scale=2.0)

    def test_region_uses_full_image_size(self):
        low = make_content(360, 640, seed=1)
        full = nearest(low, 1280, 720)
        self.check(full, roi=(200, 100, 1000, 600), method="nearest", scale=2.0)

    def test_near_native_1_07(self):
        base = make_paint(560, 461)
        shot = np.asarray(base.resize((600, 494), Image.BILINEAR))
        self.check(shot, scale=600 / 560, tol=0.03)

    def test_bilinear_1_5(self):
        self.check(bilinear(self.content, 1920, 1080), method="smooth", scale=1.5)

    def test_native_is_not_invented(self):
        self.check(make_content(1080, 1920, seed=7), method="none", scale=1.0)

    def test_nested_system_2x_and_render_1_5(self):
        low = make_content(360, 640, seed=11)
        mid = nearest(low, 960, 540)
        shot = nearest(mid, 1920, 1080)
        self.check(shot, method="nearest", scale=3.0, tol=0.02)

    def test_letterbox_is_cropped(self):
        low = make_content(360, 640, seed=14)
        shot = bilinear(low, 1280, 720)
        framed = np.zeros((840, 1280, 3), np.uint8)
        framed[60:780] = shot
        py, nat = self.check(framed)
        self.assertEqual(nat["bars"], list(py.bars))
        self.assertEqual(nat["content_box"], list(py.content_box))

    def test_real_demo_screenshot(self):
        path = os.path.join(ROOT, "docs", "demo.png")
        if not os.path.exists(path):
            self.skipTest("docs/demo.png is not available")
        arr = np.asarray(Image.open(path).convert("RGB"))
        self.check(arr, method="smooth", scale=1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
