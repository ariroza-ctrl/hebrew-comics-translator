"""
png2ico.py -- convert an image (PNG/JPG) into a Windows app_icon.ico

Usage:
    python png2ico.py my_logo.png

Creates app_icon.ico next to this script, with all the sizes Windows
uses (16/24/32/48/64/128/256 px). Best input: a SQUARE image, at least
256x256 pixels.
"""

import sys
from pathlib import Path

from PIL import Image

SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python png2ico.py <image.png>")
        sys.exit(1)

    src = Path(sys.argv[1])
    if not src.is_file():
        print(f"File not found: {src}")
        sys.exit(1)

    img = Image.open(src).convert("RGBA")

    # Pad to a square so the icon isn't distorted
    side = max(img.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))

    out = Path(__file__).parent / "app_icon.ico"
    canvas.save(out, format="ICO", sizes=SIZES)
    print(f"Created: {out}")
    print("Now run build_exe.bat - the icon will be baked into the exe.")


if __name__ == "__main__":
    main()
