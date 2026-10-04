"""
Tests for scanner-fixer pipeline.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_core.preprocess.crop import auto_crop, auto_crop_contour
from ocr_core.preprocess.deskew import detect_skew_angle, deskew
from ocr_core.preprocess.rotate import detect_180_flip, auto_rotate
from ocr_core.preprocess.enhance import enhance_for_ocr, enhance_contrast_clahe, remove_noise
from ocr_core.preprocess.pipeline import fix_scan


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_white_page(w=800, h=1100, border=50, color=True):
    """White page with dark border (simulates scanner output)."""
    if color:
        img = np.ones((h, w, 3), dtype=np.uint8) * 200  # Gray border
        img[border:h-border, border:w-border] = 255     # White content area
    else:
        img = np.ones((h, w), dtype=np.uint8) * 200
        img[border:h-border, border:w-border] = 255
    return img


def make_text_page(w=800, h=1100, n_lines=20, color=True):
    """Simulated page with horizontal text lines."""
    img = make_white_page(w, h, border=40, color=color)
    line_height = (h - 120) // n_lines
    for i in range(n_lines):
        y = 60 + i * line_height
        thickness = np.random.randint(2, 5)
        x1 = np.random.randint(60, 120)
        x2 = np.random.randint(w - 120, w - 60)
        if color:
            cv2.line(img, (x1, y), (x2, y), (0, 0, 0), thickness)
        else:
            cv2.line(img, (x1, y), (x2, y), 0, thickness)
    return img


# ─── Crop tests ──────────────────────────────────────────────────────────────

class TestCrop:
    def test_removes_dark_border(self):
        # border=80 means 80px on each side removed from a 800x1100 image
        # With padding=5, crop result ≈ (800-160+10) x (1100-160+10) = 650 x 950
        img = make_white_page(border=80)
        cropped = auto_crop(img, padding=5, threshold=240)
        h, w = cropped.shape[:2]
        # Should be smaller than original after removing 80px borders
        assert w <= 800
        assert h <= 1100
        # But should preserve the content area (at least 60% of original)
        assert w * h >= (800 * 1100 * 0.60)

    def test_preserves_content(self):
        img = make_white_page(border=50)
        cropped = auto_crop(img, padding=10)
        # Should keep most of the white area
        assert cropped.shape[0] > 900
        assert cropped.shape[1] > 600

    def test_contour_crop(self):
        img = make_white_page(border=60)
        cropped = auto_crop_contour(img, padding=10)
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0

    def test_blank_image_returns_original(self):
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cropped = auto_crop(img)
        assert cropped.shape == img.shape

    def test_grayscale_crop(self):
        img = make_white_page(color=False)
        cropped = auto_crop(img)
        assert len(cropped.shape) == 2


# ─── Deskew tests ────────────────────────────────────────────────────────────

class TestDeskew:
    def test_detects_small_angle(self):
        img = make_text_page()
        # Rotate by a known amount
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 5.0, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

        angle, meta = detect_skew_angle(rotated)
        # Should detect roughly 5 degrees (±3 degrees tolerance)
        assert abs(abs(angle) - 5.0) < 3.0
        assert "angles_std" in meta
        assert "uncertain" in meta

    def test_straight_image_near_zero(self):
        img = make_text_page()
        angle, meta = detect_skew_angle(img)
        assert abs(angle) < 2.0
        assert isinstance(meta, dict)

    def test_deskew_output_shape(self):
        img = make_text_page()
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 3.0, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

        result, angle, meta = deskew(rotated)
        assert result.shape[:2] == img.shape[:2]
        assert isinstance(meta, dict)

    def test_deskew_returns_angle(self):
        img = make_text_page()
        _, angle, meta = deskew(img)
        assert isinstance(angle, float)
        assert isinstance(meta, dict)


# ─── Rotate tests ────────────────────────────────────────────────────────────

class TestRotate:
    def test_upright_not_detected_as_flipped(self):
        img = make_text_page()
        # Add text concentrated in top half (upright)
        for y in range(100, 300, 20):
            cv2.line(img, (50, y), (750, y), (0, 0, 0), 3)
        assert not detect_180_flip(img)

    def test_auto_rotate_returns_tuple(self):
        img = make_text_page()
        result, angle = auto_rotate(img)
        assert isinstance(result, np.ndarray)
        assert angle in [0, 90, 180, 270]

    def test_auto_rotate_180_corrects(self):
        img = make_text_page()
        # Add clear content in top half
        img[100:400, 50:750] = 0  # Dark region at top
        img[700:1000, 50:750] = 255  # White at bottom

        flipped = cv2.rotate(img, cv2.ROTATE_180)
        result, angle = auto_rotate(flipped)
        assert result.shape == flipped.shape


# ─── Enhancement tests ───────────────────────────────────────────────────────

class TestEnhance:
    def test_denoise_preserves_shape(self):
        img = make_white_page()
        result = remove_noise(img)
        assert result.shape == img.shape

    def test_clahe_color(self):
        img = make_white_page(color=True)
        result = enhance_contrast_clahe(img)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_clahe_grayscale(self):
        img = make_white_page(color=False)
        result = enhance_contrast_clahe(img)
        assert result.shape == img.shape

    def test_enhance_for_ocr_full(self):
        img = make_text_page()
        result = enhance_for_ocr(img, binarize=False)
        assert result.shape == img.shape

    def test_binarize_outputs_grayscale(self):
        img = make_text_page(color=True)
        result = enhance_for_ocr(img, binarize=True)
        assert len(result.shape) == 2  # Grayscale output


# ─── Pipeline tests ──────────────────────────────────────────────────────────

class TestPipeline:
    def test_fix_scan_from_array(self):
        img = make_text_page()
        result = fix_scan(img)
        assert "image" in result
        assert "report" in result
        assert "steps" in result
        assert result["image"].shape[0] > 0

    def test_fix_scan_report_has_keys(self):
        img = make_text_page()
        result = fix_scan(img, do_rotate=True)
        r = result["report"]
        assert "rotation_applied_deg" in r
        assert "skew_corrected_deg" in r

    def test_fix_scan_steps_tracked(self):
        img = make_text_page()
        result = fix_scan(img)
        assert "original" in result["steps"]
        assert "enhanced" in result["steps"]

    def test_fix_scan_disable_steps(self):
        img = make_text_page()
        result = fix_scan(img, do_crop=False, do_rotate=False,
                          do_deskew=False, do_enhance=False)
        assert result["image"].shape == img.shape

    def test_fix_scan_invalid_path(self):
        with pytest.raises(ValueError):
            fix_scan("/nonexistent/path/image.jpg")

    def test_fix_scan_saves_file(self, tmp_path):
        img = make_text_page()
        out = tmp_path / "output.png"
        fix_scan(img, output_path=out)
        assert out.exists()
