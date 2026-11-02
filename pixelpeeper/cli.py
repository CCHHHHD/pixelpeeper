"""Command line interface: analyse one or more screenshots."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

from PIL import Image

from . import __version__
from .analysis import analyse_image, analyse_region, load_image
from .report import build_card


def _print_result(res, path: str):
    name = os.path.basename(path)
    print()
    print(f"■ {name}")
    print(f"  截图      {res.width} × {res.height}")
    if res.method == "none":
        print(f"  结果      未发现放大网格（{res.summary()}）")
    else:
        kind = "像素级放大" if res.method == "nearest" else "平滑放大"
        print(f"  推断      ≈ {res.internal_w} × {res.internal_h}   "
              f"{res.percent:.1f}% · {res.scale:.3f}× · {kind}")
    print(f"  置信度    {res.confidence * 100:.0f}%")
    if res.candidates and res.method != "none":
        tops = " / ".join(f"{c.width}×{c.height}" for c in res.candidates[:3])
        print(f"  常见档位  {tops}")
    if res.notes:
        print(f"  说明      {res.notes[0]}")
    print(f"  耗时      {res.elapsed:.1f}s")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        prog="pixelpeeper",
        description="从游戏截图中推断内部渲染分辨率（像素网格 / 重采样分析）",
    )
    ap.add_argument("images", nargs="+", help="截图文件（PNG / JPG / WebP ...）")
    ap.add_argument("--version", action="version", version=f"pixelpeeper {__version__}")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--save-card", metavar="DIR", help="为每张截图保存证据图到目录")
    ap.add_argument("--roi", metavar="X,Y,W,H",
                    help="只分析这个矩形选区，但仍按整图分辨率换算")
    ap.add_argument("--quick", action="store_true", help="跳过逐块分析，速度更快")
    args = ap.parse_args(argv)

    results = []
    failed = 0
    roi_box = None
    if args.roi:
        try:
            x, y, w, h = (int(v) for v in args.roi.replace(" ", "").split(","))
            roi_box = (x, y, x + w, y + h)
        except Exception:
            print("--roi 格式应为 X,Y,W,H", file=sys.stderr)
            return 2
    for path in args.images:
        try:
            if roi_box is not None:
                res = analyse_region(load_image(path), roi_box, path=path, quick=args.quick)
            else:
                res = analyse_image(path, quick=args.quick)
            results.append(res)
            if args.json:
                continue
            _print_result(res, path)
            if args.save_card:
                os.makedirs(args.save_card, exist_ok=True)
                stem = os.path.splitext(os.path.basename(path))[0]
                out = os.path.join(args.save_card, f"{stem}_数毛报告.png")
                build_card(res, Image.open(path).convert("RGB"), os.path.basename(path)).save(out)
                print(f"  证据图    {out}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"\n■ {path}\n  分析失败：{exc}", file=sys.stderr)

    if args.json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    elif len(results) > 1:
        print(f"\n完成 {len(results)} 张，失败 {failed} 张。")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
