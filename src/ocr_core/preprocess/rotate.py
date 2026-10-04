"""
rotate.py
Detects and corrects 180° flipped scans (upside-down pages).

Strategy:
- Text in most languages (Arabic, English) has more ink/content
  in the upper portion of each text line.
- A flipped page will have heavier weight in the wrong half.
- We use vertical projection analysis + optional OSD (Tesseract).
"""

import cv2
import numpy as np
from typing import Tuple


def detect_180_flip(image: np.ndarray) -> bool:
    """
    Detects if a scanned page is rotated 180° (upside-down).

    Uses two complementary methods:
    1. Vertical center-of-mass: text tends to sit in upper portion of lines
    2. Top-half vs bottom-half content density comparison

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        True if image appears to be flipped 180°
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # Threshold to get ink as white pixels
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Method 1: Center-of-mass in vertical direction
    # For upright text, content should be distributed somewhat evenly
    # but slightly more in the upper region due to ascenders
    rows_sum = np.sum(thresh, axis=1).astype(float)

    if rows_sum.sum() == 0:
        return False  # Blank page

    # Weighted center of mass (normalized row index)
    row_indices = np.arange(h)
    center_of_mass = float(np.sum(row_indices * rows_sum) / np.sum(rows_sum))
    center_ratio = center_of_mass / h  # 0 = top, 1 = bottom

    # Method 2: Top vs bottom half ink density
    top_half = thresh[:h // 2, :]
    bottom_half = thresh[h // 2:, :]
    top_density = float(np.sum(top_half))
    bottom_density = float(np.sum(bottom_half))

    # For Arabic text specifically: check for more descenders below midline
    # This is a heuristic — for mixed documents use Tesseract OSD instead
    flipped_votes = 0

    # Vote 1: center of mass in lower 55% suggests flipped
    if center_ratio > 0.55:
        flipped_votes += 1

    # Vote 2: bottom half significantly denser (>15% more)
    if bottom_density > top_density * 1.15:
        flipped_votes += 1

    # Method 3: Line-by-line analysis
    # In upright text, within each text line the ascenders are at top
    # We analyze the density profile in small horizontal bands
    band_size = max(5, h // 40)
    band_densities = []
    for start in range(0, h - band_size, band_size):
        band = thresh[start:start + band_size, :]
        band_densities.append(float(np.sum(band)))

    # Find the text region (bands with significant content)
    mean_density = np.mean(band_densities)
    text_bands = [d for d in band_densities if d > mean_density * 0.3]

    if len(text_bands) > 4:
        # Compare first quarter vs last quarter of text bands
        q = len(text_bands) // 4
        first_q = np.mean(text_bands[:q])
        last_q = np.mean(text_bands[-q:])

        # If last quarter is much heavier, might be flipped
        if last_q > first_q * 1.2:
            flipped_votes += 1

    return flipped_votes >= 2


def auto_rotate(image: np.ndarray, use_tesseract: bool = False) -> Tuple[np.ndarray, int]:
    """
    Automatically corrects 180° rotation in scanned pages.

    Args:
        image: Input image (BGR or grayscale)
        use_tesseract: If True, uses Tesseract OSD for more accurate detection
                       (requires pytesseract installed)

    Returns:
        Tuple of (corrected image, rotation applied in degrees: 0 or 180)
    """
    if use_tesseract:
        result = _detect_with_tesseract(image)
        if result is not None:
            angle, rotated = result
            return rotated, angle

    # Fall back to heuristic method
    is_flipped = detect_180_flip(image)

    if is_flipped:
        corrected = cv2.rotate(image, cv2.ROTATE_180)
        return corrected, 180
    else:
        return image, 0


def _detect_with_tesseract(image: np.ndarray):
    """
    Uses Tesseract OSD to detect page orientation.
    Returns (angle, rotated_image) or None if Tesseract unavailable.
    """
    try:
        import pytesseract
        from PIL import Image

        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        osd = pytesseract.image_to_osd(pil_image, output_type=pytesseract.Output.DICT)
        rotation = int(osd.get("rotate", 0))
        confidence = float(osd.get("orientation_conf", 0))

        # Only apply if confidence is reasonable
        if confidence < 1.0:
            return None

        if rotation == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
            return 180, rotated
        elif rotation == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return 90, rotated
        elif rotation == 270:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            return 270, rotated
        else:
            return 0, image

    except (ImportError, Exception):
        return None
