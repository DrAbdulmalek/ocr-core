"""
auto_crop.py
Removes dark scanner borders and excess whitespace around the document.
Works for both color and grayscale scans.
"""

import cv2
import numpy as np


def auto_crop(image: np.ndarray, padding: int = 10, threshold: int = 240) -> np.ndarray:
    """
    Automatically crops dark scanner borders and excess whitespace.

    Args:
        image: Input image (BGR or grayscale numpy array)
        padding: Pixels to keep around detected content (default 10)
        threshold: Pixel value above which is considered "white/background" (0-255)

    Returns:
        Cropped image
    """
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert so content is white on black background
    # Then threshold to find content regions
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Remove small noise via morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find bounding box of all content
    coords = cv2.findNonZero(thresh)

    if coords is None:
        # No content found, return original
        return image

    x, y, w, h = cv2.boundingRect(coords)

    # Add padding (clipped to image bounds)
    height, width = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    return image[y1:y2, x1:x2]


def auto_crop_contour(image: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Alternative crop using largest contour detection.
    Better for scans with colored backgrounds or dark borders.

    Args:
        image: Input image (BGR numpy array)
        padding: Pixels of extra space around detected document

    Returns:
        Cropped image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Take the largest contour (the document)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Sanity check: contour must cover at least 20% of image
    height, width = image.shape[:2]
    if w * h < 0.2 * width * height:
        return auto_crop(image, padding)  # Fall back to threshold method

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    return image[y1:y2, x1:x2]
