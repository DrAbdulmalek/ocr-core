"""
enhance.py
Prepares scanned images for OCR by:
- Denoising
- Contrast enhancement (CLAHE)
- Binarization (optional)
- DPI normalization

Handles both color and grayscale scans.
"""

import cv2
import numpy as np
from typing import Optional


def enhance_for_ocr(
    image: np.ndarray,
    target_dpi: Optional[int] = None,
    source_dpi: Optional[int] = None,
    denoise: bool = True,
    enhance_contrast: bool = True,
    binarize: bool = False,
    sharpen: bool = True
) -> np.ndarray:
    """
    Full enhancement pipeline for OCR preparation.

    Args:
        image: Input image (BGR color or grayscale)
        target_dpi: Target DPI for OCR (300 recommended). Requires source_dpi.
        source_dpi: Source DPI of the scan (e.g. 150, 200, 600)
        denoise: Apply denoising filter
        enhance_contrast: Apply CLAHE contrast enhancement
        binarize: Convert to binary black/white (best for pure text, bad for forms/images)
        sharpen: Apply sharpening filter

    Returns:
        Enhanced image ready for OCR
    """
    result = image.copy()

    # Step 1: DPI normalization (upscale if needed)
    if target_dpi and source_dpi and source_dpi != target_dpi:
        result = normalize_dpi(result, source_dpi, target_dpi)

    # Step 2: Denoise
    if denoise:
        result = remove_noise(result)

    # Step 3: Contrast enhancement
    if enhance_contrast:
        result = enhance_contrast_clahe(result)

    # Step 4: Sharpen
    if sharpen:
        result = sharpen_image(result)

    # Step 5: Binarize (optional — only for pure text documents)
    if binarize:
        result = adaptive_binarize(result)

    return result


def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes scanner noise while preserving text edges.
    Uses Non-Local Means denoising — better quality than Gaussian blur.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        # Color image: use colored NLM denoising
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=5,           # Filter strength for luminance (3-10)
            hColor=5,      # Filter strength for color (3-10)
            templateWindowSize=7,
            searchWindowSize=21
        )
    else:
        # Grayscale
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h=5,
            templateWindowSize=7,
            searchWindowSize=21
        )


def enhance_contrast_clahe(image: np.ndarray) -> np.ndarray:
    """
    Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Much better than global histogram equalization — avoids over-brightening.

    For color images, applies CLAHE only to the luminance channel (LAB color space)
    to avoid color distortion.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if len(image.shape) == 3:
        # Convert to LAB, enhance L channel only, convert back
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """
    Applies mild sharpening to improve text edge clarity.
    Uses unsharp masking — more controlled than simple kernel sharpening.

    Args:
        image: Input image

    Returns:
        Sharpened image
    """
    # Unsharp mask: sharp = original + (original - blurred) * amount
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


def adaptive_binarize(image: np.ndarray) -> np.ndarray:
    """
    Converts to clean black-and-white using adaptive thresholding.
    Handles uneven illumination (common in scanner edges).

    Best for: pure text documents, handwriting
    Avoid for: forms with colored fields, images, stamps

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Binary image (grayscale, 0 or 255)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Adaptive threshold handles lighting variations across the page
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,   # Size of neighborhood area (must be odd)
        C=10            # Constant subtracted from mean
    )

    return binary


def normalize_dpi(image: np.ndarray, source_dpi: int, target_dpi: int) -> np.ndarray:
    """
    Resamples image to target DPI.
    OCR works best at 300 DPI. Upscale if below, downscale if above.

    Args:
        image: Input image
        source_dpi: Current DPI of the scan
        target_dpi: Target DPI for OCR

    Returns:
        Resampled image
    """
    if source_dpi == target_dpi:
        return image

    scale = target_dpi / source_dpi
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Use LANCZOS for downscaling, CUBIC for upscaling
    if scale < 1.0:
        interpolation = cv2.INTER_LANCZOS4
    else:
        interpolation = cv2.INTER_CUBIC

    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def get_estimated_dpi(image: np.ndarray) -> Optional[int]:
    """
    Rough DPI estimation based on image size.
    Assumes typical A4 paper (210 x 297 mm).
    Returns None if estimation is uncertain.

    Args:
        image: Input image

    Returns:
        Estimated DPI or None
    """
    h, w = image.shape[:2]

    # A4 at common DPIs:
    # 150 DPI: 1240 x 1754
    # 200 DPI: 1654 x 2338
    # 300 DPI: 2480 x 3508
    # 600 DPI: 4960 x 7016

    common_dpis = {
        150: (1240, 1754),
        200: (1654, 2338),
        300: (2480, 3508),
        600: (4960, 7016),
    }

    # Use the longer dimension for comparison
    long_side = max(w, h)
    best_match = None
    best_diff = float("inf")

    for dpi, (short, long) in common_dpis.items():
        diff = abs(long_side - long)
        if diff < best_diff:
            best_diff = diff
            best_match = dpi

    # Only return if reasonably close (within 20%)
    if best_match and best_diff < long_side * 0.20:
        return best_match

    return None


from typing import Optional
