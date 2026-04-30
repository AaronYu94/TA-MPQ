"""Render poster.html to a 40 in x 30 in landscape PDF via Playwright.

Usage:
    python poster/render_poster.py [output_path]

Default output is poster/poster.pdf.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    src = Path(__file__).resolve().parent / "poster.html"
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else src.with_suffix(".pdf")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 3840, "height": 2880})
        page.goto(src.as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(out),
            width="40in",
            height="30in",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            prefer_css_page_size=True,
        )
        browser.close()

    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
