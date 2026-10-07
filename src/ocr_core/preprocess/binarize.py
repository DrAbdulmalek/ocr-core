"""
binarize.py
Document binarization algorithms implemented from their published
definitions (no proprietary code):

- Otsu (1979): global threshold maximizing between-class variance.
  Reference: N. Otsu, "A Threshold Selection Method from Gray-Level
  Histograms", IEEE Trans. SMC 9(1), 1979.
- Sauvola & Pietikäinen (2000): adaptive local threshold from the mean
  and standard deviation computed over a sliding window, using integral
  images for O(w*h) cost independent of the window size.
  Reference: J. Sauvola, M. Pietikäinen, "Adaptive document image
  binarization", Pattern Recognition 33(2), 2000.
- Auto router: picks global vs adaptive by two independent page
  measurements.
    1. illumination uniformity (low-frequency luminance spread) —
       camera photos with shadows/gradients → Sauvola. Encodes the
       practical rule in docs/09 §2.2: global Otsu collapses on
       non-uniform lighting.
    2. ink contrast (Otsu class-mean separation) — faded/grey ink on
       light paper → Sauvola. Encodes the gap measured on the golden
       sample (docs/GOLDEN-SAMPLE.md §6.3): auto routed low-contrast
       pages to Otsu at a 6× CER penalty because uniformity only sees
       lighting, not ink.

Clean high-contrast flatbed scans still go to Otsu, which is cleaner
on paper texture and never bleeds background into strokes.

Output convention everywhere: text = black (0), background = white (255),
i.e. ready for engines that expect black-on-white input.

Core math is pure NumPy; OpenCV is used only (optionally) for the BGR→gray
conversion when available.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

try:  # optional — the math below never requires cv2
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

__all__ = [
    "to_grayscale",
    "otsu_threshold",
    "binarize_otsu",
    "binarize_sauvola",
    "illumination_uniformity",
    "ink_contrast",
    "binarize_auto",
]


# ─── Helpers ────────────────────────────────────────────────────────────────


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/color or grayscale input to uint8 grayscale.

    Uses cv2 when available; otherwise falls back to the ITU-R BT.601
    luma coefficients (0.299 R + 0.587 G + 0.114 B).
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0].astype(np.uint8, copy=False)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected 2D grayscale or 3-channel image, got shape {arr.shape}")
    if cv2 is not None:
        return cv2.cvtColor(arr.astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY)
    # BT.601 luma on the BGR channel order used elsewhere in this package
    luma = (
        0.114 * arr[:, :, 0].astype(np.float64)
        + 0.587 * arr[:, :, 1].astype(np.float64)
        + 0.299 * arr[:, :, 2].astype(np.float64)
    )
    return np.clip(np.round(luma), 0, 255).astype(np.uint8)


# ─── Otsu (1979) — global threshold ─────────────────────────────────────────


def otsu_threshold(gray: np.ndarray) -> int:
    """Return the Otsu threshold (0..255) for a grayscale image.

    Vectorized over all 256 candidate thresholds using cumulative sums of
    the histogram: for each t, maximize
        sigma_B^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2
    where w0/w1 are class probabilities and mu0/mu1 class means.

    Degenerate inputs (uniform image, empty classes for every t) fall back
    to the luminance mean as the threshold, which keeps the function total
    (never raises) on blank pages.
    """
    g = to_grayscale(gray)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0

    # cumulative class-0 weight and first moment up to t (inclusive)
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * np.arange(256, dtype=np.float64))
    total_mean = m0[-1]

    w1 = total - w0
    # avoid division by zero: only thresholds with both classes non-empty
    valid = (w0 > 0) & (w1 > 0)
    if not np.any(valid):
        # uniform image — every pixel identical; mean is a valid split point
        return int(round(total_mean / total)) if total else 0

    mu_total = total_mean / total
    mu0 = np.zeros(256, dtype=np.float64)
    mu1 = np.zeros(256, dtype=np.float64)
    mu0[valid] = m0[valid] / w0[valid]
    mu1[valid] = (total_mean - m0[valid]) / w1[valid]

    sigma_b = np.zeros(256, dtype=np.float64)
    sigma_b[valid] = w0[valid] * w1[valid] * (mu0[valid] - mu1[valid]) ** 2

    # argmax over between-class variance; ties resolved to the lower
    # threshold (matches the classic formulation and keeps output stable)
    t = int(np.argmax(sigma_b))
    return t


def binarize_otsu(
    image: np.ndarray, return_threshold: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Global Otsu binarization: pixels <= t become text (black/0)."""
    g = to_grayscale(image)
    t = otsu_threshold(g)
    binary = np.where(g <= t, 0, 255).astype(np.uint8)
    if return_threshold:
        return binary, t
    return binary


# ─── Sauvola (2000) — adaptive local threshold via integral images ──────────


def _integral_sums(g: np.ndarray, pad: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return integral arrays of the edge-replicated image and its square.

    Replication padding means border windows keep their full w*w count,
    matching the standard "clamped window" semantics of adaptive
    thresholding implementations.
    """
    padded = np.pad(g, pad_width=pad, mode="edge")
    s = np.cumsum(np.cumsum(padded, axis=0, dtype=np.float64), axis=1)
    s2 = np.cumsum(np.cumsum(padded.astype(np.float64) ** 2, axis=0), axis=1)
    # prepend a zero row/column so window sums become plain subtractions
    s = np.pad(s, ((1, 0), (1, 0)))
    s2 = np.pad(s2, ((1, 0), (1, 0)))
    return s, s2


def sauvola_thresholds(
    gray: np.ndarray, window: int = 31, k: float = 0.2
) -> np.ndarray:
    """Compute the per-pixel Sauvola threshold map for a grayscale image.

        t(x, y) = m(x, y) * (1 + k * (s(x, y) / R - 1))

    with m/s the window mean/standard deviation, R = 128 (half of the
    8-bit dynamic range) and k the bias towards the mean (typical
    document range 0.2–0.34; lower k preserves low-contrast strokes).

    Cost is O(h*w) regardless of window size thanks to integral images,
    which is what makes Sauvola practical at 300 DPI page sizes.
    """
    if window < 3:
        raise ValueError("window must be >= 3")
    if window % 2 == 0:
        window += 1  # enforce odd so the pixel sits in the window center
    g = to_grayscale(gray)
    h, w = g.shape
    half = window // 2

    s, s2 = _integral_sums(g, half)

    # window sums for every pixel: rectangle between (y-half, x-half) and
    # (y+half, x+half) inclusive → indices into the padded integral arrays
    y0 = np.arange(h)
    x0 = np.arange(w)
    r0 = y0                 # top row index (inclusive) in integral coords
    r1 = y0 + window        # bottom row index (exclusive)
    c0 = x0
    c1 = x0 + window

    area = float(window * window)
    # broadcast: sum over rectangle = s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0]
    sum_ = s[np.ix_(r1, c1)] - s[np.ix_(r0, c1)] - s[np.ix_(r1, c0)] + s[np.ix_(r0, c0)]
    sum2 = s2[np.ix_(r1, c1)] - s2[np.ix_(r0, c1)] - s2[np.ix_(r1, c0)] + s2[np.ix_(r0, c0)]

    mean = sum_ / area
    var = np.clip(sum2 / area - mean ** 2, 0.0, None)
    std = np.sqrt(var)

    R = 128.0
    thr = mean * (1.0 + k * (std / R - 1.0))
    return thr


def binarize_sauvola(
    image: np.ndarray,
    window: int = 31,
    k: float = 0.2,
    return_threshold: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Adaptive Sauvola binarization: pixel < local threshold → text (0)."""
    g = to_grayscale(image)
    thr = sauvola_thresholds(g, window=window, k=k)
    binary = np.where(g.astype(np.float64) <= thr, 0, 255).astype(np.uint8)
    if return_threshold:
        return binary, thr
    return binary


# ─── Page measurements: illumination + ink contrast ─────────────────────────


def illumination_uniformity(gray: np.ndarray, blocks: int = 8) -> float:
    """Measure low-frequency illumination spread of a page.

    Splits the (grayscale) image into a `blocks × blocks` grid and takes
    the **80th percentile** of each block's luminance — a background
    estimate that is robust to ink density. (Using the block *mean* makes
    the metric conflate text density with lighting: a dense paragraph
    block and a margin block differ by ~15% even on a perfectly flat
    scan. The 80th percentile sits on the paper, not the ink.)

    Returns the coefficient of variation (std/mean) of the block
    statistics: ~0.00–0.05 → flat scan; > ~0.12 → visible shadows or
    lighting gradients.
    """
    g = to_grayscale(gray).astype(np.float64)
    h, w = g.shape
    if h < blocks or w < blocks:
        return 0.0
    block_h = h // blocks
    block_w = w // blocks
    stats = []
    for by in range(blocks):
        for bx in range(blocks):
            patch = g[
                by * block_h: (by + 1) * block_h,
                bx * block_w: (bx + 1) * block_w,
            ]
            stats.append(np.percentile(patch, 80))
    stats_arr = np.asarray(stats, dtype=np.float64)
    overall = stats_arr.mean()
    if overall <= 0:
        return 0.0
    return float(stats_arr.std() / overall)


def ink_contrast(gray: np.ndarray) -> float:
    """Measure foreground/background luminance separation.

    Applies Otsu's split and returns the normalized distance between the
    two class means:

        (μ_paper − μ_ink) / 255

    Dark ink on white paper scores ~0.7–0.9. Faded/grey ink on light
    paper (the golden-sample `low_contrast` condition: ink=118, paper=238)
    scores ~0.4–0.5. A uniform page scores 0.

    This is independent of illumination_uniformity: a faded-ink scan can
    be perfectly flat-lit and still need Sauvola, which is exactly the
    6× CER gap measured in docs/GOLDEN-SAMPLE.md §6.3.
    """
    g = to_grayscale(gray)
    t = otsu_threshold(g)
    ink = g[g <= t]
    paper = g[g > t]
    if ink.size == 0 or paper.size == 0:
        return 0.0
    return float((paper.mean() - ink.mean()) / 255.0)


# ─── Auto router: global vs adaptive by illumination + ink contrast ─────────


def binarize_auto(
    image: np.ndarray,
    uniformity_limit: float = 0.12,
    contrast_limit: float = 0.55,
    sauvola_window: int = 31,
    sauvola_k: float = 0.2,
) -> Dict[str, object]:
    """Route each page to the right binarization family automatically.

    Decision rule (docs/09 §2.2 + golden-sample §6.3 encoded):
      - illumination_uniformity > uniformity_limit → Sauvola (adaptive),
        because a global threshold collapses on shadows/gradients;
      - else if ink_contrast < contrast_limit → Sauvola (adaptive),
        because faded ink is a local-std problem Otsu's single cut
        cannot recover (6× CER penalty measured on the golden sample);
      - otherwise → Otsu (global), which is cleaner on high-contrast
        flat scans and never bleeds background texture into strokes.

    Returns a dict rather than a bare array so callers (and tests) can
    log *why* a page was routed the way it was:

        {"image": uint8 binary, "method": "otsu"|"sauvola",
         "illumination_ratio": float, "ink_contrast": float,
         "threshold": int|None}
    """
    g = to_grayscale(image)
    ratio = illumination_uniformity(g)
    contrast = ink_contrast(g)
    if ratio > uniformity_limit or contrast < contrast_limit:
        binary, thr = binarize_sauvola(
            g, window=sauvola_window, k=sauvola_k, return_threshold=True
        )
        return {
            "image": binary,
            "method": "sauvola",
            "illumination_ratio": ratio,
            "ink_contrast": contrast,
            "threshold": None,  # per-pixel map, not a single value
        }
    binary, thr = binarize_otsu(g, return_threshold=True)
    return {
        "image": binary,
        "method": "otsu",
        "illumination_ratio": ratio,
        "ink_contrast": contrast,
        "threshold": int(thr),
    }
