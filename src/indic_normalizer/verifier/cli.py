"""``indic-verifier [RUN_DIR] [--audio-dir DIR] [--data-dir DIR] [--port N] [--no-browser]``"""

from __future__ import annotations

import argparse
from typing import List, Optional


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dir", nargs="?", help="OUT_DIR/<data dir> of a verbalize run (or choose it in the page)")
    p.add_argument("--audio-dir", help="folder with <stem>.wav (or choose it in the page)")
    p.add_argument("--data-dir", help="the input folder verbalize read (needed to change decided numbers)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")


def run(a: argparse.Namespace) -> int:
    from .server import serve
    serve(a.run_dir, a.audio_dir, a.port, open_browser=not a.no_browser, data_dir=getattr(a, "data_dir", None))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="indic-verifier",
                                description="check numbers and fix sentences of a verbalize run in the browser "
                                            "(no GPU; Windows / Linux)")
    add_arguments(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
