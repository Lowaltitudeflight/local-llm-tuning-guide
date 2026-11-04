"""Render the repository social preview (1280x640, GitHub's recommended size).

Run: python -I tools/make_social_preview.py
Output: docs/images/social-preview.png

Requires Pillow (pip install pillow). CI does not run this script, so the
dependency is optional and lives outside the documented toolchain.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
S = 2  # supersample

BG = (11, 15, 23)
GRID = (24, 31, 45)
TEXT = (232, 238, 246)
MUTED = (138, 152, 172)
ACCENT = (94, 234, 170)      # green
ACCENT2 = (122, 162, 255)    # blue
WARN = (255, 196, 92)        # amber

FONT_DIR = r"C:\Windows\Fonts"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size * S)


BOLD = "arialbd.ttf"
REG = "arial.ttf"
MONO = "consola.ttf"

DATA = {
    "c": [1, 4, 8, 16, 32, 64],
    "short": [35.8, 107.7, 71.7, 242.1, 350.7, 372.0],
    "long": [34.1, 101.3, 131.5, 184.7, 211.9, 218.9],
}

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "images", "social-preview.png",
)


def px(v: float) -> int:
    return int(round(v * S))


def main() -> None:
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)

    # subtle grid
    for x in range(0, W, 40):
        d.line([(px(x), 0), (px(x), px(H))], fill=GRID, width=1)
    for y in range(0, H, 40):
        d.line([(0, px(y)), (px(W), px(y))], fill=GRID, width=1)

    # left accent bar
    d.rectangle([0, 0, px(6), px(H)], fill=ACCENT)

    # ---------------- left column ----------------
    x = 68
    d.text((px(x), px(64)), "LOCAL LLM TUNING GUIDE", font=font(BOLD, 17), fill=ACCENT)
    d.text((px(x), px(104)), "No NVLink.", font=font(BOLD, 62), fill=TEXT)
    d.text((px(x), px(176)), "Still 372 tok/s.", font=font(BOLD, 62), fill=TEXT)
    d.text(
        (px(x), px(256)),
        "Evidence-graded decisions for 8x72 GB PCIe boxes",
        font=font(REG, 22),
        fill=MUTED,
    )

    stats = [
        ("TP8 vs TP4", "+12% at 2x the cards", ACCENT),
        ("476 GiB weights", "will not fit 4x72 GB", WARN),
        ("KV cache", "785 B/token, measured", ACCENT2),
    ]
    y = 316
    for label, value, color in stats:
        d.rectangle([px(x), px(y + 6), px(x + 4), px(y + 34)], fill=color)
        d.text((px(x + 18), px(y)), label, font=font(BOLD, 20), fill=TEXT)
        d.text((px(x + 18), px(y + 26)), value, font=font(REG, 18), fill=MUTED)
        y += 62

    d.text(
        (px(x), px(H - 62)),
        "github.com/Lowaltitudeflight/local-llm-tuning-guide",
        font=font(MONO, 16),
        fill=MUTED,
    )

    # ---------------- right column: chart ----------------
    cx0, cy0, cx1, cy1 = 690, 150, 1210, 520
    d.text((px(cx0), px(96)), "DeepSeek-V4.1-Flash  |  8x RTX PRO 5000 72GB, TP8", font=font(REG, 15), fill=MUTED)
    d.text((px(cx0), px(120)), "output tok/s vs concurrency (3 launches, mean)", font=font(REG, 14), fill=(100, 112, 130))

    # axes
    d.line([(px(cx0), px(cy1)), (px(cx1), px(cy1))], fill=(52, 62, 80), width=px(1))
    d.line([(px(cx0), px(cy0)), (px(cx0), px(cy1))], fill=(52, 62, 80), width=px(1))

    vmax = 400.0
    n = len(DATA["c"])

    def xy(i: int, v: float) -> tuple[int, int]:
        xf = cx0 + (i / (n - 1)) * (cx1 - cx0)
        yf = cy1 - (v / vmax) * (cy1 - cy0)
        return px(xf), px(yf)

    # horizontal gridlines + labels
    for val in (0, 100, 200, 300, 400):
        _, yy = xy(0, val)
        d.line([(px(cx0), yy), (px(cx1), yy)], fill=(30, 38, 54), width=px(0.6))
        d.text((px(cx0) - px(42), yy - px(9)), str(val), font=font(MONO, 13), fill=(96, 108, 126))

    # x labels
    for i, c in enumerate(DATA["c"]):
        xx, _ = xy(i, 0)
        d.text((xx - px(10), px(cy1) + px(10)), f"c{c}", font=font(MONO, 14), fill=(96, 108, 126))

    # series
    for key, color in (("short", ACCENT2), ("long", ACCENT)):
        pts = [xy(i, v) for i, v in enumerate(DATA[key])]
        for a, b in zip(pts, pts[1:]):
            d.line([a, b], fill=color, width=px(3))
        for p in pts:
            r = px(5)
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=BG, outline=color, width=px(2.5))

    # annotate the c8 dip
    dip = xy(2, DATA["short"][2])
    d.line([dip[0], dip[1] - px(10), dip[0] + px(58), dip[1] - px(46)], fill=WARN, width=px(1.2))
    d.text((dip[0] + px(62), dip[1] - px(58)), "c8 dip, reproducible", font=font(REG, 13), fill=WARN)

    # legend
    lx, ly = cx0, cy1 + 46
    d.line([(px(lx), px(ly)), (px(lx + 26), px(ly))], fill=ACCENT2, width=px(3))
    d.text((px(lx + 34), px(ly - 9)), "short 256/128", font=font(REG, 14), fill=MUTED)
    d.line([(px(lx + 170), px(ly)), (px(lx + 196), px(ly))], fill=ACCENT, width=px(3))
    d.text((px(lx + 204), px(ly - 9)), "long 8192/256", font=font(REG, 14), fill=MUTED)

    out = img.resize((W, H), Image.LANCZOS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.save(OUT, "PNG", optimize=True)
    print("wrote", OUT, out.size)


if __name__ == "__main__":
    main()
