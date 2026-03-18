"""
Generate synthetic PCB-like sample images for testing.

These are clearly artificial — they simulate board textures with
intentional defects so the pipeline has something to chew on.
Run once: python data/generate_samples.py
"""

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def _draw_traces(draw: ImageDraw.Draw, w: int, h: int):
    """Draw horizontal and vertical 'traces' like a PCB."""
    for _ in range(12):
        y = random.randint(0, h)
        draw.line([(0, y), (w, y)], fill=(50, 120, 50), width=2)
    for _ in range(12):
        x = random.randint(0, w)
        draw.line([(x, 0), (x, h)], fill=(50, 120, 50), width=2)


def _draw_pads(draw: ImageDraw.Draw, w: int, h: int):
    """Draw circular solder pads."""
    for _ in range(20):
        cx, cy = random.randint(20, w - 20), random.randint(20, h - 20)
        r = random.randint(4, 8)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(180, 180, 100))


def create_base_board(w: int = 512, h: int = 512) -> Image.Image:
    """Create a base PCB-like image."""
    # Dark green substrate
    img = Image.new("RGB", (w, h), (20, 80, 30))
    draw = ImageDraw.Draw(img)
    _draw_traces(draw, w, h)
    _draw_pads(draw, w, h)

    # Add slight texture
    noise = np.random.randint(0, 15, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(np.clip(np.array(img) + noise, 0, 255).astype(np.uint8))
    return img


def add_scratch(img: Image.Image) -> Image.Image:
    """Add a diagonal scratch defect."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x1, y1 = random.randint(50, w // 2), random.randint(50, h // 2)
    x2, y2 = x1 + random.randint(100, 200), y1 + random.randint(80, 160)
    draw.line([(x1, y1), (x2, y2)], fill=(160, 140, 100), width=3)
    draw.line([(x1 + 2, y1 + 2), (x2 + 2, y2 + 2)], fill=(140, 120, 80), width=1)
    return img


def add_contamination(img: Image.Image) -> Image.Image:
    """Add blob-like contamination."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(3):
        cx, cy = random.randint(80, w - 80), random.randint(80, h - 80)
        rx, ry = random.randint(15, 40), random.randint(15, 40)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(90, 70, 50))
    return img.filter(ImageFilter.GaussianBlur(radius=1))


def add_solder_bridge(img: Image.Image) -> Image.Image:
    """Add a bright solder bridge between pads."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx, cy = random.randint(100, w - 100), random.randint(100, h - 100)
    draw.rectangle([cx, cy, cx + 30, cy + 8], fill=(210, 210, 180))
    # Bright glint
    draw.rectangle([cx + 5, cy + 2, cx + 25, cy + 6], fill=(240, 240, 220))
    return img


def generate_all():
    """Generate all sample images."""
    out = config.SAMPLES_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Normal board
    img = create_base_board()
    img.save(out / "sample_normal.png")
    print(f"  Saved {out / 'sample_normal.png'}")

    # Scratched board
    img = create_base_board()
    img = add_scratch(img)
    img.save(out / "sample_scratch.png")
    print(f"  Saved {out / 'sample_scratch.png'}")

    # Contaminated board
    img = create_base_board()
    img = add_contamination(img)
    img.save(out / "sample_contamination.png")
    print(f"  Saved {out / 'sample_contamination.png'}")

    # Solder bridge
    img = create_base_board()
    img = add_solder_bridge(img)
    img.save(out / "sample_solder_bridge.png")
    print(f"  Saved {out / 'sample_solder_bridge.png'}")

    print(f"\nDone — {len(list(out.glob('*.png')))} sample images in {out}")


if __name__ == "__main__":
    generate_all()
