"""Deterministic renderer for the unified golden sample.

Turns each :class:`~ocr_core.benchmarks.golden.dataset.GoldenPair` into a
grayscale PIL image — a pure function of ``(pair, font file)``:

Rendering strategy (proven empirically, 2026-10-05):
  1. PRIMARY — libraqm layout (``PIL.features.check("raqm")``): draw the raw
     logical string with ``direction="rtl", language="ar"``. Raqm/HarfBuzz
     applies real OpenType shaping, producing book-grade Arabic exactly like
     natural documents. With the bundled Noto Naskh font this yields
     CER 0.0000 on clean Arabic pages with tesseract's tessdata_fast ara —
     the pre-bidi workaround could never reach that (≈0.79, wrong output
     order: tesseract re-reads the pre-reversed visual string LTR).
  2. FALLBACK — ``arabic_reshaper`` + ``bidi`` (no raqm environments;
     documented lower quality, kept so CI without raqm still renders).

The four conditions from docs/09 §6/§2.2 are synthesised:
  clean         : flat white page, near-black ink
  shadow        : diagonal illumination gradient (the documented
                  Otsu-collapse case — global thresholding eats the dark half)
  low_contrast  : mid-grey ink on light-grey paper
  noisy         : salt & pepper + mild gaussian noise

No global RNG state: numpy's ``default_rng(pair.seed)`` only.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from .dataset import CONDITIONS, GoldenPair

__all__ = ["render_pair", "resolve_font", "raqm_available", "FONT_CANDIDATES"]

#: bundled fonts ship with the package (OFL — see fonts/OFL.txt)
_BUNDLED_FONTS_DIR = Path(__file__).parent / "fonts"

_SYSTEM_FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/freefont",
    "/usr/share/fonts/truetype/liberation",
]

_SYSTEM_CANDIDATES = {
    "ar": ["NotoNaskhArabic-Regular.ttf", "NotoNaskhArabic.ttf", "Amiri-Regular.ttf",
           "FreeSerif.ttf", "DejaVuSans.ttf"],
    "en": ["DejaVuSans.ttf", "FreeSans.ttf", "LiberationSans-Regular.ttf"],
}

#: exported for tests/docs — mapping lang → candidate font file names (system)
FONT_CANDIDATES = {k: tuple(v) for k, v in _SYSTEM_CANDIDATES.items()}

_BUNDLED = {
    "ar": "NotoNaskhArabic-Regular.ttf",
}

_SHADOW_STRENGTH = 0.55   # bottom-right multiplier floor (1.0 → 0.45)
_SALT_PEPPER_RATE = 0.004
_GAUSS_SIGMA = 6.0


def raqm_available() -> bool:
    """True when PIL can apply HarfBuzz shaping (preferred path)."""
    try:
        return bool(features.check("raqm"))
    except Exception:  # noqa: BLE001 - probe must never break rendering
        return False


def resolve_font(lang: str) -> Path:
    """Return the font file for *lang* — bundled OFL font first (deterministic)."""
    bundled = _BUNDLED.get(lang)
    if bundled and (_BUNDLED_FONTS_DIR / bundled).is_file():
        return _BUNDLED_FONTS_DIR / bundled
    for name in _SYSTEM_CANDIDATES.get(lang, _SYSTEM_CANDIDATES["en"]):
        for d in _SYSTEM_FONT_DIRS:
            p = Path(d) / name
            if p.is_file():
                return p
    # last resort: ask fontconfig (keeps the module usable on odd images)
    fc = shutil.which("fc-match")
    if fc:  # pragma: no cover - environment-dependent path
        import subprocess

        try:
            out = subprocess.run(
                [fc, "-f", "%{file}", "DejaVu Sans"], capture_output=True, text=True, timeout=10
            )
            p = Path(out.stdout.strip())
            if p.is_file():
                return p
        except Exception:  # noqa: BLE001 - probe must never break rendering
            pass
    raise FileNotFoundError(
        "no usable TrueType font found for golden-sample rendering; "
        f"searched bundled {_BUNDLED_FONTS_DIR} and {_SYSTEM_FONT_DIRS}"
    )


def _shape_arabic_legacy(line: str) -> str:
    """Legacy no-raqm path: pre-shape + pre-reverse (documented lower quality)."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as exc:  # pragma: no cover - extra not installed
        raise ImportError(
            "rendering Arabic golden pairs needs the [golden] extra "
            "(arabic-reshaper + python-bidi) when libraqm is unavailable"
        ) from exc
    return get_display(arabic_reshaper.reshape(line))


def _apply_shadow(arr: np.ndarray, seed: int) -> np.ndarray:
    h, w = arr.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]
    # diagonal gradient: bright top-left → dark bottom-right
    mask = 1.0 - _SHADOW_STRENGTH * (0.6 * yy + 0.4 * xx)
    out = arr.astype(np.float64) * mask
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _apply_noise(arr: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = arr.astype(np.float64)
    out += rng.normal(0.0, _GAUSS_SIGMA, size=arr.shape)
    salt = rng.random(arr.shape) < _SALT_PEPPER_RATE
    out[salt] = 255.0
    pepper = rng.random(arr.shape) < _SALT_PEPPER_RATE
    out[pepper] = 0.0
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def render_pair(pair: GoldenPair, font_path: Path | None = None) -> Image.Image:
    """Render *pair* to a grayscale PIL image (pure, deterministic)."""
    if pair.condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {pair.condition!r}")
    font_path = font_path or resolve_font(pair.lang)
    font = ImageFont.truetype(str(font_path), size=pair.font_px)

    use_raqm = raqm_available()
    if not use_raqm and pair.lang == "ar":
        lines = [_shape_arabic_legacy(l) for l in pair.text.split("\n")]
    else:
        lines = pair.text.split("\n")

    draw_kwargs = {}
    if use_raqm:
        draw_kwargs = {
            "direction": "rtl" if pair.lang == "ar" else "ltr",
            "language": "ar" if pair.lang == "ar" else "en",
        }

    pad = 60
    line_h = int(pair.font_px * 1.7)
    width = 1240
    height = pad * 2 + line_h * len(lines)

    # deterministic per-condition background level
    bg = {"clean": 255, "shadow": 250, "low_contrast": 238, "noisy": 252}[pair.condition]
    ink = {"clean": 25, "shadow": 25, "low_contrast": 118, "noisy": 28}[pair.condition]

    img = Image.new("L", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        text_w = draw.textlength(line, font=font, **draw_kwargs)
        # RTL lines hang from the right margin, LTR from the left
        x = max(pad, width - pad - text_w) if pair.lang == "ar" else pad
        draw.text((x, pad + i * line_h), line, font=font, fill=ink, **draw_kwargs)

    arr = np.asarray(img)
    if pair.condition == "shadow":
        arr = _apply_shadow(arr, pair.seed)
    elif pair.condition == "noisy":
        arr = _apply_noise(arr, pair.seed)
    return Image.fromarray(arr, mode="L")
