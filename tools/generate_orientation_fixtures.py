#!/usr/bin/env python3
"""Generate EXIF-orientation fixtures for native reference-image input.

Every file stores the same upright 96x64 picture under a different EXIF
orientation (1-8). Diffusers' `load_image` applies `ImageOps.exif_transpose`,
so each file must decode to the upright picture; `test-image-orientation`
checks that the native loader does the same.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

# exif_transpose applies these methods; storing the inverse makes the upright
# picture the expected decode for every orientation.
STORE_METHOD = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}


def upright() -> Image.Image:
    # Four distinct quadrant colours distinguish all eight orientations.
    image = Image.new("RGB", (96, 64))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 47, 31), fill=(220, 30, 30))
    draw.rectangle((48, 0, 95, 31), fill=(30, 200, 60))
    draw.rectangle((0, 32, 47, 63), fill=(30, 60, 220))
    draw.rectangle((48, 32, 95, 63), fill=(240, 210, 40))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("tests/fixtures/image_orientation")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    picture = upright()
    for orientation in range(1, 9):
        stored = picture if orientation == 1 else picture.transpose(STORE_METHOD[orientation])
        exif = Image.Exif()
        exif[0x0112] = orientation
        path = args.output / f"orientation_{orientation}.jpg"
        stored.save(path, quality=100, subsampling=0, exif=exif.tobytes())
        restored = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        error = max(
            abs(a - b) for a, b in zip(restored.tobytes(), picture.tobytes())
        )
        if restored.size != picture.size or error > 24:
            raise SystemExit(f"{path}: exif_transpose mismatch size={restored.size} max_error={error}")
        print(f"{path.name}: stored={stored.size} exif_transpose max_error={error}")


if __name__ == "__main__":
    main()
