"""
Tests for ocr_core.preprocess.binarize — Otsu/Sauvola/auto routing.

Algorithms are implemented from their published definitions (Otsu 1979,
Sauvola & Pietikäinen 2000); these tests verify the mathematical
properties that make them correct for document binarization, using
synthetic pages (no fixtures needed).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_core.preprocess.binarize import (
    binarize_auto,
    binarize_otsu,
    binarize_sauvola,
    illumination_uniformity,
    otsu_threshold,
    sauvola_thresholds,
    to_grayscale,
)


# ─── Synthetic page helpers ──────────────────────────────────────────────────


def _flat_text_page(w: int = 400, h: int = 300, bg: int = 235, ink: int = 30) -> np.ndarray:
    """Clean flatbed-style scan: white page, dark horizontal text strokes."""
    img = np.full((h, w), bg, dtype=np.uint8)
    for y in range(40, h - 40, 24):
        img[y: y + 4, 40: w - 40] = ink
    return img


def _shadow_text_page(w: int = 400, h: int = 300) -> np.ndarray:
    """Camera-style page: same dark text everywhere, but background has a
    strong vertical illumination gradient (bright top → dark bottom)."""
    gradient = np.linspace(240, 90, h, dtype=np.float64)[:, None]
    img = np.repeat(gradient, w, axis=1).astype(np.uint8)
    for y in range(40, h - 40, 24):
        img[y: y + 4, 40: w - 40] = 25
    return img


# ─── Otsu ────────────────────────────────────────────────────────────────────


def test_otsu_separates_bimodal_histogram():
    img = _flat_text_page(bg=235, ink=30)
    t = otsu_threshold(img)
    # every threshold on the [ink, bg) plateau is optimal; the argmax tie
    # resolves to the ink edge — the split must capture all ink and
    # never swallow the background mode
    assert 30 <= t < 235


def test_otsu_binary_output_convention_text_black():
    img = _flat_text_page()
    binary = binarize_otsu(img)
    assert set(np.unique(binary)) <= {0, 255}
    assert binary[binary == 0].size > 0          # text present as black
    assert binary[0, 0] == 255                   # corner background stays white
    # background fraction dominates a sparse-text page
    assert (binary == 255).mean() > 0.7


def test_otsu_degenerate_uniform_image_does_not_crash():
    img = np.full((64, 64), 200, dtype=np.uint8)
    t = otsu_threshold(img)
    assert 0 <= t <= 255
    binary = binarize_otsu(img)
    assert binary.shape == img.shape


def test_otsu_return_threshold_flag():
    img = _flat_text_page()
    binary, t = binarize_otsu(img, return_threshold=True)
    assert binary.dtype == np.uint8
    assert isinstance(t, int)


def test_otsu_matches_bruteforce_between_class_variance():
    """Reference check: brute-force the published objective over 256
    thresholds and compare with the vectorized implementation."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(80, 80)).astype(np.uint8)
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    best_t, best_var = 0, -1.0
    for t in range(256):
        w0 = hist[: t + 1].sum()
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (hist[: t + 1] * np.arange(t + 1)).sum() / w0
        mu1 = (hist[t + 1:] * np.arange(t + 1, 256)).sum() / w1
        var = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    assert otsu_threshold(img) == best_t


# ─── Sauvola ─────────────────────────────────────────────────────────────────


def test_sauvola_preserves_text_in_shadowed_and_bright_regions():
    img = _shadow_text_page()
    binary = binarize_sauvola(img, window=31, k=0.2)
    h = img.shape[0]
    top = binary[: h // 2]
    bottom = binary[h // 2:]
    # text strokes must survive binarization in BOTH lighting regimes
    assert (top == 0).mean() > 0.005
    assert (bottom == 0).mean() > 0.005


def test_sauvola_beats_otsu_on_shadow_page_text_recovery():
    """The documented failure mode (docs/09 §2.2): a global threshold
    collapses on non-uniform lighting — the dark half of the page swallows
    its background — while Sauvola keeps a sane ink fraction everywhere."""
    img = _shadow_text_page()
    otsu_binary = binarize_otsu(img)
    sauvola_binary = binarize_sauvola(img, window=31)
    h = img.shape[0]
    otsu_dark_ratio = (otsu_binary[h // 2:] == 0).mean()
    sauvola_dark_ratio = (sauvola_binary[h // 2:] == 0).mean()
    sauvola_top_ratio = (sauvola_binary[: h // 2] == 0).mean()
    # Otsu degenerates in the dark region (background swallowed)
    assert otsu_dark_ratio > 0.5
    # Sauvola stays in the sane ink range (~1/6 strokes) in BOTH regimes
    assert 0.05 < sauvola_dark_ratio < 0.30
    assert 0.05 < sauvola_top_ratio < 0.30


def test_sauvola_even_window_is_corrected_not_rejected():
    img = _flat_text_page(w=120, h=90)
    out_even = binarize_sauvola(img, window=30)
    out_odd = binarize_sauvola(img, window=31)
    assert np.array_equal(out_even, out_odd)


def test_sauvola_invalid_window_raises():
    with pytest.raises(ValueError):
        sauvola_thresholds(np.zeros((20, 20), dtype=np.uint8), window=2)


def test_sauvola_threshold_map_shape_and_range():
    img = _flat_text_page(w=150, h=100)
    thr = sauvola_thresholds(img, window=15)
    assert thr.shape == img.shape
    assert np.all(np.isfinite(thr))


# ─── Illumination uniformity + auto router ───────────────────────────────────


def test_uniformity_flat_near_zero_gradient_higher():
    flat = illumination_uniformity(_flat_text_page())
    shadow = illumination_uniformity(_shadow_text_page())
    assert flat < 0.05
    assert shadow > 2 * flat
    assert shadow > 0.12  # above the default routing limit


def test_auto_routes_flat_scan_to_otsu():
    result = binarize_auto(_flat_text_page())
    assert result["method"] == "otsu"
    assert isinstance(result["threshold"], int)
    assert result["threshold"] is None or 0 <= result["threshold"] <= 255


def test_auto_routes_shadowed_page_to_sauvola():
    result = binarize_auto(_shadow_text_page())
    assert result["method"] == "sauvola"
    assert result["threshold"] is None  # per-pixel map, not one number


def test_auto_metadata_shape():
    result = binarize_auto(_flat_text_page())
    assert {"image", "method", "illumination_ratio", "threshold"} == set(result.keys())
    assert 0.0 <= result["illumination_ratio"] <= 1.0
    assert result["image"].dtype == np.uint8


# ─── Grayscale conversion + pipeline wiring ──────────────────────────────────


def test_to_grayscale_accepts_color_and_gray():
    color = np.stack([_flat_text_page()] * 3, axis=-1)  # BGR
    gray = to_grayscale(color)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8
    assert np.array_equal(gray, _flat_text_page())


def test_apply_binarization_dispatch_and_legacy_default():
    from ocr_core.preprocess.enhance import apply_binarization, enhance_for_ocr

    img = _flat_text_page()
    for method in ("adaptive", "otsu", "sauvola", "auto"):
        out = apply_binarization(img, method)
        assert out.shape == img.shape
        assert set(np.unique(out)) <= {0, 255}
    with pytest.raises(ValueError):
        apply_binarization(img, "magic")

    # legacy path unchanged: default binarize_method == "adaptive"
    legacy = enhance_for_ocr(img, denoise=False, enhance_contrast=False,
                             sharpen=False, binarize=True)
    direct = apply_binarization(img, "adaptive")
    assert np.array_equal(legacy, direct)


def test_fix_scan_accepts_binarize_method_passthrough():
    from ocr_core.preprocess.pipeline import fix_scan

    result = fix_scan(
        _flat_text_page(w=600, h=800),
        do_crop=False, do_deskew=False,
        do_enhance=True, binarize=True, binarize_method="auto",
    )
    assert set(np.unique(result["image"])) <= {0, 255}
