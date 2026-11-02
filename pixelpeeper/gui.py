"""Tkinter desktop UI for PixelPeeper."""
from __future__ import annotations

import ctypes
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageGrab, ImageOps, ImageTk

from .analysis import AnalysisResult, analyse_array
from .dnd import DropTarget
from . import __version__
from .report import build_card, build_icon, font as pil_font, mini_chart

APP_TITLE = "数毛器 · PixelPeeper"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".tga"}

BG2 = "#0b0f14"
CARD = "#141a21"
CARD2 = "#19202a"
LINE = "#26303b"
TXT = "#e8eef4"
MUT = "#8a96a3"
GRN = "#48d6ab"
AMB = "#f0b25a"
RED = "#ee6e6e"


class Preview(tk.Canvas):
    def __init__(self, master, app):
        super().__init__(master, bg="#0d1218", highlightthickness=0, cursor="crosshair")
        self.app = app
        self.pil: Image.Image | None = None
        self.result: AnalysisResult | None = None
        self.scale = 1.0
        self.zoom = 2.0
        self.cx = 0.0
        self.cy = 0.0
        self.fit = True
        self.show_grid = tk.BooleanVar(value=True)
        self.select_mode = tk.BooleanVar(value=True)
        self.roi: tuple[int, int, int, int] | None = None
        self._img_ref = None
        self._view = (1.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0))
        self._drag = None
        self._roi_start = None
        self._selecting = False
        self.bind("<Configure>", lambda e: self.render())
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_move)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<ButtonPress-3>", self._on_pan_press)
        self.bind("<B3-Motion>", self._on_pan_move)
        self.bind("<ButtonRelease-3>", lambda e: setattr(self, "_drag", None))
        self.bind("<Double-Button-1>", self._toggle_zoom)

    # ------------------------------------------------------------------ #
    def set_image(self, pil: Image.Image, result: AnalysisResult | None = None):
        self.pil = pil
        self.result = result
        self.roi = result.roi if result is not None else None
        self.fit = True
        self.cx, self.cy = pil.width / 2, pil.height / 2
        self.show_grid.set(result is not None and result.has_grid)
        self.render()

    def set_result(self, result: AnalysisResult):
        self.result = result
        if result.roi is not None:
            self.roi = result.roi
        self.show_grid.set(result.has_grid)
        self.render()

    # ------------------------------------------------------------------ #
    def _fit_scale(self) -> float:
        if not self.pil:
            return 1.0
        cw, ch = max(10, self.winfo_width()), max(10, self.winfo_height())
        return min((cw - 28) / self.pil.width, (ch - 28) / self.pil.height, 4.0)

    def _on_wheel(self, event):
        if not self.pil:
            return
        factor = 1.18 if event.delta > 0 else 1 / 1.18
        base = self._fit_scale() if self.fit else self.scale
        self.scale = max(self._fit_scale() * 0.75, min(16.0, base * factor))
        self.fit = False
        self.render()

    def _toggle_zoom(self, _event=None):
        if not self.pil:
            return
        if self.select_mode.get():
            return
        if self.fit:
            self.scale = max(2.0, self._fit_scale() * 2)
            self.fit = False
        else:
            self.fit = True
        self.render()

    def _to_image(self, x: float, y: float) -> tuple[float, float]:
        s, ox, oy, box = self._view
        ix = box[0] + (x - ox) / s
        iy = box[1] + (y - oy) / s
        return ix, iy

    def _clamp_image(self, ix: float, iy: float) -> tuple[float, float]:
        assert self.pil is not None
        return (max(0.0, min(ix, self.pil.width - 1.0)),
                max(0.0, min(iy, self.pil.height - 1.0)))

    def _on_press(self, event):
        if not self.pil:
            self.app.choose_file()
            return
        if self.select_mode.get():
            ix, iy = self._clamp_image(*self._to_image(event.x, event.y))
            self._roi_start = (ix, iy)
            self.roi = None
            self._selecting = True
            self.configure(cursor="tcross")
            self.app.on_roi_changed()
            self.render()
            return
        self._drag = (event.x, event.y, self.cx, self.cy)
        self.configure(cursor="fleur")

    def _on_move(self, event):
        if not self.pil:
            return
        if self._selecting and self._roi_start:
            ax, ay = self._roi_start
            bx, by = self._clamp_image(*self._to_image(event.x, event.y))
            x0, x1 = sorted((ax, bx))
            y0, y1 = sorted((ay, by))
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                self.roi = (int(round(x0)), int(round(y0)),
                            int(round(x1)), int(round(y1)))
                self.render()
            return
        if not self._drag:
            return
        x0, y0, cx0, cy0 = self._drag
        s = self.scale
        self.cx = cx0 - (event.x - x0) / s
        self.cy = cy0 - (event.y - y0) / s
        self.fit = False
        self.render()

    def _on_release(self, _event):
        if self._selecting:
            self._selecting = False
            self._roi_start = None
            self.configure(cursor="crosshair")
            if self.roi and (self.roi[2] - self.roi[0] < 8 or self.roi[3] - self.roi[1] < 8):
                self.roi = None
            self.app.on_roi_changed()
            self.render()
            return
        self._drag = None

    def _on_pan_press(self, event):
        if not self.pil:
            return
        self._drag = (event.x, event.y, self.cx, self.cy)
        self.configure(cursor="fleur")

    def _on_pan_move(self, event):
        self._on_move(event)

    # ------------------------------------------------------------------ #
    def render(self):
        self.delete("all")
        cw, ch = max(10, self.winfo_width()), max(10, self.winfo_height())
        if not self.pil:
            self._empty_state(cw, ch)
            return
        iw, ih = self.pil.size
        base = self._fit_scale() if self.fit else self.scale
        self.scale = max(0.05, min(16.0, base))
        s = self.scale
        vw, vh = cw / s, ch / s
        vw, vh = min(vw, iw), min(vh, ih)
        self.cx = min(max(self.cx, vw / 2), iw - vw / 2) if vw < iw else iw / 2
        self.cy = min(max(self.cy, vh / 2), ih - vh / 2) if vh < ih else ih / 2
        box = (self.cx - vw / 2, self.cy - vh / 2, self.cx + vw / 2, self.cy + vh / 2)
        crop = self.pil.crop(tuple(int(round(v)) for v in box))
        size = (max(1, int(round(crop.width * s))), max(1, int(round(crop.height * s))))
        resample = Image.NEAREST if s >= 1 else Image.LANCZOS
        shown = crop.resize(size, resample)
        if self.show_grid.get() and self.result and self.result.has_grid and s >= 1.6:
            shown = self._with_grid(shown, box, s)
        self._img_ref = ImageTk.PhotoImage(shown)
        ox = (cw - size[0]) // 2
        oy = (ch - size[1]) // 2
        self.create_image(ox, oy, image=self._img_ref, anchor="nw")
        self.create_rectangle(ox - 1, oy - 1, ox + size[0], oy + size[1],
                              outline="#26333f", width=1)
        self._view = (s, ox, oy, box)
        self._draw_roi(cw, ch)
        # corner HUD
        text = f"{s * 100:.0f}%  ·  {int(vw)}×{int(vh)} px"
        self.create_text(cw - 14, ch - 12, text=text, anchor="se", fill="#6f7d8c", font=("Microsoft YaHei UI", 10))

    def _draw_roi(self, cw: int, ch: int):
        if not self.roi or not self.pil:
            return
        s, ox, oy, box = self._view
        x0 = (self.roi[0] - box[0]) * s + ox
        y0 = (self.roi[1] - box[1]) * s + oy
        x1 = (self.roi[2] - box[0]) * s + ox
        y1 = (self.roi[3] - box[1]) * s + oy
        # dim outside the ROI so the analysed area stands out
        for rect in ((0, 0, cw, y0), (0, y1, cw, ch), (0, y0, x0, y1), (x1, y0, cw, y1)):
            if rect[2] > rect[0] and rect[3] > rect[1]:
                self.create_rectangle(*rect, fill="#000000", outline="", stipple="gray50")
        self.create_rectangle(x0, y0, x1, y1, outline=AMB, width=2, dash=(7, 4))
        label = f"选区 {self.roi[2]-self.roi[0]}×{self.roi[3]-self.roi[1]}  ·  {self.roi[0]},{self.roi[1]}"
        tx, ty = x0 + 6, max(12, y0 - 10)
        tw = len(label) * 7 + 12
        self.create_rectangle(tx - 4, ty - 10, tx + tw, ty + 8, fill="#111820", outline=AMB, width=1)
        self.create_text(tx + 2, ty - 1, text=label, anchor="w", fill=AMB,
                         font=("Microsoft YaHei UI", 10, "bold"))

    def _with_grid(self, shown: Image.Image, box, s: float) -> Image.Image:
        res = self.result
        assert res is not None
        step = res.scale_x * s
        if step < 5:
            return shown
        rx0 = res.roi[0] if res.roi else 0
        ry0 = res.roi[1] if res.roi else 0
        ph = (rx0 + (res.evidence_x.phase or 0.0)) % max(1e-6, res.scale_x)
        ov = Image.new("RGBA", shown.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        x = ((ph - box[0]) % res.scale_x) * s
        while x < shown.width:
            d.line([(x, 0), (x, shown.height)], fill=(72, 214, 171, 40), width=1)
            x += step
        phy = (ry0 + (res.evidence_y.phase or 0.0)) % max(1e-6, res.scale_y)
        y = ((phy - box[1]) % res.scale_y) * s
        while y < shown.height:
            d.line([(0, y), (shown.width, y)], fill=(72, 214, 171, 22), width=1)
            y += step
        return Image.alpha_composite(shown.convert("RGBA"), ov).convert("RGB")

    def _empty_state(self, cw, ch):
        self.create_rectangle(18, 18, cw - 18, ch - 18, outline="#31404f", width=2, dash=(7, 6))
        self.create_text(cw / 2, ch / 2 - 78, text="把游戏截图拖进来", fill=TXT,
                         font=("Microsoft YaHei UI", 30, "bold"))
        self.create_text(cw / 2, ch / 2 - 30, text="或者 Ctrl+V 粘贴 · 点击这里打开文件",
                         fill=MUT, font=("Microsoft YaHei UI", 15))
        self.create_text(cw / 2, ch / 2 + 16, text="支持 PNG / JPG / WebP / BMP", fill="#6b7a89",
                         font=("Microsoft YaHei UI", 12))
        self.create_text(cw / 2, ch / 2 + 78, text="建议使用无损截图（PNG），并尽量避开 HUD 和字幕",
                         fill="#6b7a89", font=("Microsoft YaHei UI", 12))


class SidebarPane(tk.Frame):
    """A vertically scrollable right-hand column."""

    def __init__(self, master, width=500):
        super().__init__(master, bg=BG2, width=width)
        self.pack_propagate(False)
        self.canvas = tk.Canvas(self, bg=BG2, highlightthickness=0, bd=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                  style="Dark.Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")
        self.inner = tk.Frame(self.canvas, bg=BG2)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.refresh())
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self._wheel_bound: set[tk.Widget] = set()

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self._win, width=event.width)
        self.refresh()

    def refresh(self):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        need = self.inner.winfo_reqheight() > self.canvas.winfo_height() + 2
        if need and not self.vbar.winfo_ismapped():
            self.vbar.pack(side="right", fill="y")
        elif not need and self.vbar.winfo_ismapped():
            self.vbar.pack_forget()

    def _wheel(self, event):
        if self.inner.winfo_reqheight() <= self.canvas.winfo_height() + 2:
            return
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def bind_wheel(self):
        def rec(w):
            if w not in self._wheel_bound:
                w.bind("<MouseWheel>", self._wheel)
                self._wheel_bound.add(w)
            for c in w.winfo_children():
                rec(c)
        rec(self.inner)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        try:
            dpi = ctypes.windll.user32.GetDpiForSystem()
            if dpi:
                self.tk.call("tk", "scaling", dpi / 72.0)
        except Exception:
            pass
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1520, sw - 80), min(960, sh - 100)
        self.geometry(f"{w}x{h}+40+20")
        self.minsize(min(1200, w), min(780, h))
        self.configure(bg=BG2)
        self._set_dpi()
        self.q: "queue.Queue" = queue.Queue()
        self.pil: Image.Image | None = None
        self.result: AnalysisResult | None = None
        self._busy = False
        self._mini_ref = None
        self._icon_ref = None
        self._build_style()
        self._build_header()
        self._build_body()
        self._build_status()
        self.dnd = DropTarget(self)
        try:
            icon = build_icon()
            self._icon_ref = ImageTk.PhotoImage(icon.resize((64, 64), Image.NEAREST))
            self.iconphoto(True, self._icon_ref)
        except Exception:
            pass
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Control-v>", lambda e: self.paste())
        self.bind("<Control-V>", lambda e: self.paste())
        self.bind("<Control-o>", lambda e: self.choose_file())
        self.after(80, self._poll)
        self.status("把截图拖进窗口，或按 Ctrl+V 粘贴", kind="muted")

    # ------------------------------------------------------------------ #
    def _set_dpi(self):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("P.Horizontal.TProgressbar", troughcolor="#1a222c", background=GRN,
                        bordercolor="#1a222c", lightcolor=GRN, darkcolor=GRN, thickness=3)
        style.configure("Dark.Vertical.TScrollbar", background="#2b3745", troughcolor=BG2,
                        bordercolor=BG2, arrowcolor="#9aa7b5",
                        darkcolor="#2b3745", lightcolor="#2b3745", relief="flat")
        style.map("Dark.Vertical.TScrollbar", background=[("active", "#3a4a5c")])

    def _build_header(self):
        bar = tk.Frame(self, bg=BG2, height=112)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        logo = tk.Canvas(bar, width=44, height=44, bg=BG2, highlightthickness=0)
        logo.pack(side="left", padx=(24, 14), pady=32)
        try:
            ic = build_icon().resize((44, 44), Image.NEAREST)
            self._logo_ref = ImageTk.PhotoImage(ic)
            logo.create_image(22, 22, image=self._logo_ref)
        except Exception:
            pass
        t = tk.Frame(bar, bg=BG2)
        t.pack(side="left", pady=22)
        tk.Label(t, text="数毛器", bg=BG2, fg=TXT, font=("Microsoft YaHei UI", 24, "bold")).pack(anchor="w")
        tk.Label(t, text=f"PIXELPEEPER · v{__version__} · 截图像素网格分析", bg=BG2, fg="#8fa0b3",
                 font=("Microsoft YaHei UI", 11)).pack(anchor="w", pady=(3, 0))

        right = tk.Frame(bar, bg=BG2)
        right.pack(side="right", padx=24, pady=32)
        self._button(right, "帮助", self.show_help, kind="ghost").pack(side="right", padx=5)
        self._button(right, "保存报告", self.save_report, kind="ghost").pack(side="right", padx=5)
        self._button(right, "粘贴", self.paste, kind="ghost").pack(side="right", padx=5)
        self._button(right, "打开图片", self.choose_file, kind="primary").pack(side="right", padx=(5, 12))

    def _button(self, parent, text, cmd, kind="ghost"):
        if kind == "primary":
            bg, fg, active = GRN, "#06231a", "#5ee6bd"
        else:
            bg, fg, active = CARD2, TXT, "#222c37"
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, activebackground=active,
                      activeforeground=fg, relief="flat", bd=0, padx=18, pady=9,
                      font=("Microsoft YaHei UI", 12, "bold" if kind == "primary" else "normal"),
                      cursor="hand2", highlightthickness=0)
        return b

    def _build_body(self):
        body = tk.Frame(self, bg=BG2)
        body.pack(fill="both", expand=True, padx=(22, 22), pady=(0, 10))
        left = tk.Frame(body, bg=BG2)
        left.pack(side="left", fill="both", expand=True)
        self.preview = Preview(left, self)
        self.preview.pack(fill="both", expand=True)

        tools = tk.Frame(left, bg=BG2, height=40)
        tools.pack(fill="x", pady=(8, 0))
        self._small(tools, "−", lambda: self.preview._on_wheel(type("E", (), {"delta": -1})()), 34)
        self._small(tools, "+", lambda: self.preview._on_wheel(type("E", (), {"delta": 1})()), 34)
        self._small(tools, "适应", self._fit, 56)
        self._small(tools, "1:1", self._one_to_one, 56)
        self.select_toggle = tk.Checkbutton(
            tools, text="框选区域", variable=self.preview.select_mode,
            command=self.on_select_mode, bg=BG2, fg=MUT, selectcolor=CARD,
            activebackground=BG2, activeforeground=TXT,
            font=("Microsoft YaHei UI", 11), bd=0, highlightthickness=0, cursor="hand2")
        self.select_toggle.pack(side="left", padx=(12, 4))
        self.btn_roi = self._action(tools, "分析选区", self.analyze_roi, primary=True)
        self.btn_roi.pack(side="left", padx=3)
        self.btn_roi.configure(state="disabled")
        self._action(tools, "分析整图", self.analyze_full).pack(side="left", padx=3)
        self._action(tools, "清除选区", self.clear_roi).pack(side="left", padx=3)
        self.grid_toggle = tk.Checkbutton(
            tools, text="源像素网格", variable=self.preview.show_grid, command=self.preview.render,
            bg=BG2, fg=MUT, selectcolor=CARD, activebackground=BG2, activeforeground=TXT,
            font=("Microsoft YaHei UI", 10), bd=0, highlightthickness=0, cursor="hand2")
        self.grid_toggle.pack(side="left", padx=(10, 0))
        tk.Label(tools, text="右键拖动平移", bg=BG2, fg="#5f6d7c",
                 font=("Microsoft YaHei UI", 9)).pack(side="right")

        self.sidepane = SidebarPane(body, 500)
        self.sidepane.pack(side="right", fill="y", padx=(20, 0))
        self.sidebar = self.sidepane.inner
        self._build_result_card(self.sidebar)
        self._build_candidates(self.sidebar)
        self._build_notes(self.sidebar)
        self._build_mini(self.sidebar)
        self.sidepane.bind_wheel()
        self.sidepane.after_idle(self.sidepane.refresh)

    def _action(self, parent, text, cmd, primary=False):
        bg = GRN if primary else CARD
        fg = "#06231a" if primary else TXT
        active = "#5ee6bd" if primary else CARD2
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                         activebackground=active, activeforeground=fg,
                         disabledforeground="#53606d", relief="flat", bd=0,
                         padx=10, pady=4, font=("Microsoft YaHei UI", 10, "bold" if primary else "normal"),
                         cursor="hand2", highlightthickness=0)

    def _small(self, parent, text, cmd, width):
        b = tk.Button(parent, text=text, command=cmd, bg=CARD, fg=MUT, activebackground=CARD2,
                      activeforeground=TXT, relief="flat", bd=0, width=0, padx=0, pady=5,
                      font=("Microsoft YaHei UI", 11), cursor="hand2", highlightthickness=0)
        b.configure(width=max(2, width // 16))
        b.pack(side="left", padx=2)
        return b

    def _card(self, parent, **kw):
        f = tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1, bd=0, **kw)
        return f

    def _build_result_card(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(0, 10))
        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill="x", padx=22, pady=16)
        top = tk.Frame(inner, bg=CARD)
        top.pack(fill="x")
        tk.Label(top, text="推断内部分辨率", bg=CARD, fg=MUT, font=("Microsoft YaHei UI", 12)).pack(side="left")
        self.conf_pill = tk.Label(top, text="等待", bg="#2a3540", fg=TXT, padx=10, pady=3,
                                  font=("Microsoft YaHei UI", 10, "bold"))
        self.conf_pill.pack(side="right")
        self.big = tk.Label(inner, text="—", bg=CARD, fg=TXT, font=("Microsoft YaHei UI", 34, "bold"))
        self.big.pack(anchor="w", pady=(4, 0))
        self.sub = tk.Label(inner, text="拖入截图开始分析", bg=CARD, fg=MUT,
                            font=("Microsoft YaHei UI", 14))
        self.sub.pack(anchor="w")
        self.method = tk.Label(inner, text="", bg=CARD, fg=GRN, font=("Microsoft YaHei UI", 12, "bold"))
        self.method.pack(anchor="w", pady=(8, 0))
        sep = tk.Frame(inner, bg=LINE, height=1)
        sep.pack(fill="x", pady=12)
        self.stats = tk.Frame(inner, bg=CARD)
        self.stats.pack(fill="x")
        self._stat_rows = []
        for label in ("截图", "内容区域", "横向 / 纵向", "耗时"):
            row = tk.Frame(self.stats, bg=CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=CARD, fg=MUT, width=10, anchor="w",
                     font=("Microsoft YaHei UI", 11)).pack(side="left")
            val = tk.Label(row, text="—", bg=CARD, fg=TXT, anchor="w",
                           font=("Microsoft YaHei UI", 11))
            val.pack(side="left", fill="x", expand=True)
            self._stat_rows.append(val)

    def _build_candidates(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(0, 12))
        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill="x", padx=22, pady=14)
        head = tk.Frame(inner, bg=CARD)
        head.pack(fill="x")
        tk.Label(head, text="常见档位", bg=CARD, fg=MUT, font=("Microsoft YaHei UI", 12)).pack(side="left")
        tk.Label(head, text="按接近程度排序", bg=CARD, fg="#5f6d7c",
                 font=("Microsoft YaHei UI", 10)).pack(side="right")
        self.cands = tk.Frame(inner, bg=CARD)
        self.cands.pack(fill="x", pady=(8, 0))

    def _build_notes(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(0, 12))
        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill="x", padx=22, pady=14)
        tk.Label(inner, text="说明", bg=CARD, fg=MUT, font=("Microsoft YaHei UI", 12)).pack(anchor="w")
        self.notes = tk.Label(inner, text="HUD/字幕通常以原生分辨率绘制，分析会自动避开画面边缘。",
                              bg=CARD, fg="#a4b0bd", justify="left", anchor="w", wraplength=440,
                              font=("Microsoft YaHei UI", 11))
        self.notes.pack(anchor="w", fill="x", pady=(6, 0))

    def _build_mini(self, parent):
        card = self._card(parent)
        card.pack(fill="x")
        self.mini = tk.Label(card, bg=CARD, bd=0)
        self.mini.pack(fill="x", padx=1, pady=1)
        self._mini_placeholder()

    def _mini_placeholder(self):
        im = Image.new("RGB", (468, 122), (20, 26, 33))
        d = ImageDraw.Draw(im)
        d.text((16, 16), "证据图", font=pil_font(13, True), fill=(160, 172, 184))
        d.text((16, 44), "分析完成后显示像素间距 / 拟合误差曲线", font=pil_font(11), fill=(100, 112, 126))
        self._mini_ref = ImageTk.PhotoImage(im)
        self.mini.configure(image=self._mini_ref)

    def _build_status(self):
        bar = tk.Frame(self, bg=BG2, height=36)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_lbl = tk.Label(bar, text="", bg=BG2, fg=MUT, font=("Microsoft YaHei UI", 11), anchor="w")
        self.status_lbl.pack(side="left", padx=24)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160,
                                        style="P.Horizontal.TProgressbar")
        tk.Label(bar, text="拖入 · Ctrl+V · Ctrl+O", bg=BG2, fg="#5f6d7c",
                 font=("Microsoft YaHei UI", 10)).pack(side="right", padx=24)

    # ------------------------------------------------------------------ #
    def status(self, text, kind="normal"):
        col = {"muted": "#6a7787", "ok": GRN, "warn": AMB, "err": RED}.get(kind, TXT)
        self.status_lbl.configure(text=text, fg=col)

    def _set_busy(self, busy, text="正在数毛…"):
        self._busy = busy
        if busy:
            self.progress.pack(side="right", padx=(0, 10))
            self.progress.start(12)
            self.status(text, "muted")
            self.big.configure(text="分析中…", fg=MUT)
            self.sub.configure(text="正在测量像素重复与重采样网格")
            self.conf_pill.configure(text="…", bg="#2a3540", fg=MUT)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "ok":
                    _, path, pil, res = msg
                    self._apply(path, pil, res)
                elif msg[0] == "err":
                    _, path, err = msg
                    self._set_busy(False)
                    self.status(f"分析失败：{err}", "err")
                    messagebox.showerror("分析失败", str(err), parent=self)
        except queue.Empty:
            pass
        if getattr(self, "dnd", None):
            try:
                paths = self.dnd.queue.get_nowait()
                if paths:
                    self.open_path(paths[0])
            except queue.Empty:
                pass
        self.after(80, self._poll)

    # ------------------------------------------------------------------ #
    def choose_file(self):
        path = filedialog.askopenfilename(
            parent=self, title="选择游戏截图",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("所有文件", "*.*")])
        if path:
            self.open_path(path)

    def paste(self):
        try:
            data = ImageGrab.grabclipboard()
        except Exception as exc:
            self.status(f"读取剪贴板失败：{exc}", "err")
            return
        if data is None:
            self.status("剪贴板里没有图片", "warn")
            return
        if isinstance(data, Image.Image):
            im = ImageOps.exif_transpose(data)
            if im.mode != "RGB":
                im = im.convert("RGB")
            self._load_image(im, "剪贴板截图")
        elif isinstance(data, list) and data:
            self.open_path(data[0])
        else:
            self.status("剪贴板内容不是图片", "warn")

    def open_path(self, path: str):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            self.status("文件不存在", "err")
            return
        if self._busy:
            self.status("正在分析上一张，请稍等…", "warn")
            return
        try:
            im = Image.open(path)
            im.load()
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self.status(f"打开失败：{exc}", "err")
            messagebox.showerror("打开失败", str(exc), parent=self)
            return
        self._load_image(im, os.path.basename(path), path=path)

    def _load_image(self, im: Image.Image, name: str, path: str = ""):
        """Show the image and wait for the user to select a region or analyse all."""
        self.pil = im
        self.result = None
        self._current_name = name
        self.preview.set_image(im, None)
        self.preview.select_mode.set(True)
        self.on_select_mode()
        self.on_roi_changed()
        self.big.configure(text=f"{im.width} × {im.height}", fg=TXT)
        self.sub.configure(text="先在画面上框选 3D 区域，再点“分析选区”", fg=MUT)
        self.method.configure(text="", fg=GRN)
        self.conf_pill.configure(text="待分析", bg="#2a3540", fg=MUT)
        for lbl in self._stat_rows:
            lbl.configure(text="—")
        for w in self.cands.winfo_children():
            w.destroy()
        tk.Label(self.cands, text="等待框选分析…", bg=CARD, fg="#6b7a89",
                 font=("Microsoft YaHei UI", 11), anchor="w").pack(fill="x", pady=6)
        self.notes.configure(text="框选时避开 HUD、字幕、UI 和大面积纯色贴图；也可以直接点“分析整图”。")
        self.status(f"已载入 {name}（{im.width}×{im.height}）· 拖拽框选 3D 区域，或点“分析整图”", "muted")

    def on_select_mode(self):
        if not hasattr(self, "select_toggle"):
            return
        self.preview.configure(cursor="crosshair" if self.preview.select_mode.get() else "fleur")

    def on_roi_changed(self, announce: bool = True):
        if not hasattr(self, "btn_roi"):
            return
        has = self.preview.roi is not None
        self.btn_roi.configure(state="normal" if has else "disabled")
        if has and announce:
            x0, y0, x1, y1 = self.preview.roi
            self.status(f"已框选 {x1-x0}×{y1-y0} @ ({x0},{y0}) · 点“分析选区”", "muted")

    def clear_roi(self):
        self.preview.roi = None
        self.preview.render()
        self.on_roi_changed()
        self.status("已清除选区", "muted")

    def analyze_full(self):
        if not self.pil or self._busy:
            return
        arr = np.asarray(self.pil)
        self._analyse(arr, "整图", roi=None, base_size=self.pil.size)

    def analyze_roi(self):
        if not self.pil or self._busy:
            return
        if not self.preview.roi:
            self.status("先在画面上拖拽框选一个区域", "warn")
            return
        x0, y0, x1, y1 = self.preview.roi
        arr = np.asarray(self.pil)[y0:y1, x0:x1]
        self._analyse(arr, "选区", roi=(x0, y0, x1, y1), base_size=self.pil.size)

    def _analyse(self, arr: np.ndarray, label: str, *,
                 roi: tuple[int, int, int, int] | None,
                 base_size: tuple[int, int]):
        self._set_busy(True, "正在分析选区…" if roi else "正在分析整图…")
        self.status(f"正在分析{label} …")
        name = getattr(self, "_current_name", "")

        def work():
            try:
                res = analyse_array(arr, path=name, base_size=base_size, roi=roi)
                self.q.put(("ok", name, self.pil.copy(), res))
            except Exception as exc:  # noqa: BLE001
                self.q.put(("err", name, exc))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ #
    def _apply(self, path: str, pil: Image.Image, res: AnalysisResult):
        self._set_busy(False)
        self.pil, self.result = pil, res
        self.preview.pil = pil
        self.preview.set_result(res)
        self._update_sidebar(path, res)
        self.status(f"完成 · {res.summary()}", "ok" if res.method != "none" else "muted")
        self.on_roi_changed(announce=False)

    def _update_sidebar(self, path: str, res: AnalysisResult):
        if res.method == "none":
            self.big.configure(text=f"{res.width} × {res.height}", fg=MUT)
            self.sub.configure(text="未发现放大网格", fg=MUT)
            self.method.configure(text="原生 / 时域重建", fg=AMB)
            self.conf_pill.configure(text=f"置信度 {round(res.confidence*100)}%", bg="#3a3324", fg=AMB)
        else:
            self.big.configure(text=f"{res.internal_w} × {res.internal_h}", fg=TXT)
            kind = "像素级放大" if res.method == "nearest" else "平滑放大"
            self.sub.configure(text=f"{res.percent:.1f}%  ·  {res.scale:.3f}×  ·  {kind}", fg=MUT)
            self.method.configure(text=("最近邻 / 整数缩放" if res.method == "nearest" else "双线性 / FSR 类重建"), fg=GRN)
            c = res.confidence
            col, bg = (GRN, "#12352b") if c >= 0.8 else ((AMB, "#3a3324") if c >= 0.55 else (RED, "#3a2424"))
            self.conf_pill.configure(text=f"置信度 {round(c*100)}%", bg=bg, fg=col)
        cw = res.content_box[2] - res.content_box[0]
        ch = res.content_box[3] - res.content_box[1]
        area = f"{cw} × {ch}" + ("（选区）" if res.roi else ("（已裁黑边）" if any(res.bars) else ""))
        values = [f"{res.width} × {res.height}",
                  area,
                  f"{res.scale_x:.3f}× / {res.scale_y:.3f}×",
                  f"{res.elapsed:.1f} s"]
        for lbl, val in zip(self._stat_rows, values):
            lbl.configure(text=val)
        for w in self.cands.winfo_children():
            w.destroy()
        if res.method == "none":
            tk.Label(self.cands, text="没有可靠的候选；可换一张主体更清晰的截图", bg=CARD, fg="#6b7a89",
                     font=("Microsoft YaHei UI", 11), anchor="w").pack(fill="x", pady=6)
        else:
            for i, c in enumerate(res.candidates[:4]):
                row = tk.Frame(self.cands, bg=CARD)
                row.pack(fill="x", pady=3)
                tk.Label(row, text=c.label, bg=CARD, fg=TXT if i == 0 else MUT,
                         font=("Microsoft YaHei UI", 11, "bold" if i == 0 else "normal"),
                         width=13, anchor="w").pack(side="left")
                tk.Label(row, text=f"{100 / c.scale:.1f}%", bg=CARD, fg=MUT,
                         font=("Microsoft YaHei UI", 11), width=8, anchor="e").pack(side="right")
                bar = tk.Frame(row, bg="#26303b", height=6, width=150)
                bar.pack(side="right", padx=10)
                fill = tk.Frame(bar, bg=GRN if i == 0 else "#3c6a5e", height=6,
                                width=int(150 * max(0.05, min(1.0, c.score))))
                fill.place(x=0, y=0)
        notes = "\n".join("· " + n for n in res.notes[:2]) if res.notes else "—"
        self.notes.configure(text=notes)
        try:
            mw = 496
            if hasattr(self, "sidepane"):
                mw = max(320, self.sidepane.canvas.winfo_width() - 8)
            im = mini_chart(res, mw, 120)
            self._mini_ref = ImageTk.PhotoImage(im)
            self.mini.configure(image=self._mini_ref)
        except Exception:
            pass
        if hasattr(self, "sidepane"):
            self.sidepane.bind_wheel()
            self.sidepane.after_idle(self.sidepane.refresh)

    # ------------------------------------------------------------------ #
    def save_report(self):
        if not self.result or not self.pil:
            self.status("先拖入一张截图", "warn")
            return
        stem = os.path.splitext(os.path.basename(self.result.path or "截图"))[0]
        path = filedialog.asksaveasfilename(
            parent=self, title="保存数毛报告", defaultextension=".png",
            initialfile=f"{stem}_数毛报告.png", filetypes=[("PNG 图片", "*.png")])
        if not path:
            return
        try:
            card = build_card(self.result, self.pil, os.path.basename(self.result.path or "截图"))
            card.save(path)
            self.status(f"报告已保存：{path}", "ok")
            if messagebox.askyesno("报告已保存", "要现在打开吗？", parent=self):
                os.startfile(path)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("保存失败", str(exc), parent=self)

    def copy_result(self):
        if not self.result:
            return
        text = self.result.summary() + "\n" + "\n".join(self.result.notes)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status("结果已复制到剪贴板", "ok")

    def show_help(self):
        win = tk.Toplevel(self)
        win.title("怎么用 / 原理")
        win.configure(bg=CARD)
        win.geometry("820x700")
        win.transient(self)
        txt = tk.Text(win, bg=CARD, fg=TXT, relief="flat", wrap="word", padx=28, pady=24,
                      font=("Microsoft YaHei UI", 13), insertbackground=TXT)
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", HELP_TEXT)
        txt.configure(state="disabled")

    def _fit(self):
        self.preview.fit = True
        self.preview.render()

    def _one_to_one(self):
        if not self.preview.pil:
            return
        self.preview.fit = False
        self.preview.scale = 1.0
        self.preview.render()

    def _on_close(self):
        try:
            self.dnd.close()
        except Exception:
            pass
        self.destroy()


HELP_TEXT = """数毛器 PixelPeeper

把游戏截图拖进窗口，或者 Ctrl+V 粘贴，工具会从像素层面判断游戏实际渲染的分辨率。

它是怎么数的
─────────────
截图按屏幕分辨率保存，但游戏经常以更低的分辨率渲染，再放大到屏幕上。放大后的画面会留下“网格”：

· 最近邻 / 整数缩放：整列、整行像素被复制，相邻像素完全相同；
· 双线性 / 双三次 / FSR：画面在源像素内部是平滑的，但在源像素边界处出现周期性的折点。

工具用两种方法交叉验证：
1. 统计重复像素的比例。放大 s 倍时，重复位置约占 1 - 1/s，所以 s = 1/(1-比例)；
2. 在候选网格上采样再重建画面，真实网格处重建误差会明显触底。

得到倍数 s 后，内部分辨率 = 截图分辨率 / s。

结果怎么看
─────────────
· 像素级放大：证据最强，基本可以直接相信；
· 平滑放大：双线性 / FSR 类，可能有 ±1 档误差，看“置信度”；
· 未发现网格：可能是原生渲染，也可能是 DLSS / TSR 等时域重建，或者截图被二次压缩过。

小技巧
─────────────
· 用 PNG 无损截图，JPEG 会削弱重复像素的证据；
· 尽量选主体清晰、斜线多、HUD 和字幕少的画面；
· 开启“显示源像素网格”，放大到 200% 以上，可以肉眼核对青线是否贴合台阶；
· 动态分辨率游戏不同区域倍数可能不同，报告里的“分块一致性”可以看出这一点。
"""


def main(argv: list[str] | None = None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    app = App()
    if argv:
        app.after(200, lambda: app.open_path(argv[0]))
    app.mainloop()
