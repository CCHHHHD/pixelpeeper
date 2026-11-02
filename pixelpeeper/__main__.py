"""Entry point: ``python -m pixelpeeper [image ...]``."""
from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help") and len(args) == 1:
        from .cli import main as cli_main
        return cli_main(["--help"])
    if args:
        from .cli import main as cli_main
        return cli_main(args)
    from .gui import main as gui_main
    gui_main([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
