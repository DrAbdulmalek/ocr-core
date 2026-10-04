"""
test_dedup_experiment.py
========================

Experimental validation of the normalize + phash dedup pipeline
using synthetic shifted/rotated/rescaled document images.

This test file documents the empirical behavior of the pipeline
and establishes the default Hamming threshold based on real measurements.

NOT a permanent test — kept for reference during threshold tuning.
Run manually: pytest packages/scanner_fixer/tests/test_dedup_experiment.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_core.preprocess.normalize import normalize_scanned_image
from ocr_core.preprocess.dedup import compute_image_phash, find_duplicate_clusters, export_dedup_report


def generate_synthetic_document(width=800, height=1100):
    """Generate a synthetic document with text-like rectangles."""
    img = np.ones((height, width), dtype=np.uint8) * 255
    np.random.seed(42)
    for y_line in range(100, height - 50, 40):
        line_width = np.random.randint(200, 700)
        x_start = np.random.randint(50, 150)
        cv2.rectangle(img, (x_start, y_line), (x_start + line_width, y_line + 8), 0, -1)
        for _ in range(np.random.randint(3, 8)):
            gap_x = x_start + np.random.randint(20, line_width - 20)
            gap_w = np.random.randint(15, 40)
            cv2.rectangle(img, (gap_x, y_line), (gap_x + gap_w, y_line + 8), 255, -1)
    cv2.rectangle(img, (200, 40), (600, 70), 0, -1)
    return img


def shift_image(image, dx=0, dy=0):
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (w, h),
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def rotate_image(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h),
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)


class TestShiftInvariance:
    """After normalize+phash, shifted images should match the original."""

    def setup_method(self):
        self.base = generate_synthetic_document()
        self.base_norm = normalize_scanned_image(self.base)
        self.base_phash = compute_image_phash(self.base_norm)

    @pytest.mark.parametrize("dx", [10, -10, 20, -20, 30, -30])
    def test_horizontal_shift_within_threshold(self, dx):
        shifted = shift_image(self.base, dx=dx)
        norm = normalize_scanned_image(shifted)
        phash = compute_image_phash(norm)
        dist = phash - self.base_phash
        assert dist <= 5, f"dx={dx}: Hamming={dist} > 5"

    @pytest.mark.parametrize("dy", [10, -10, 20, -20, 30, -30])
    def test_vertical_shift_within_threshold(self, dy):
        shifted = shift_image(self.base, dy=dy)
        norm = normalize_scanned_image(shifted)
        phash = compute_image_phash(norm)
        dist = phash - self.base_phash
        assert dist <= 5, f"dy={dy}: Hamming={dist} > 5"


class TestResolutionInvariance:
    """After normalize+phash, images at different DPIs should produce identical hashes."""

    def setup_method(self):
        self.base = generate_synthetic_document()
        self.base_norm = normalize_scanned_image(self.base)
        self.base_phash = compute_image_phash(self.base_norm)

    @pytest.mark.parametrize("scale", [0.5, 0.75, 1.5, 2.0])
    def test_resolution_invariant(self, scale):
        h, w = self.base.shape[:2]
        resized = cv2.resize(self.base, (int(w * scale), int(h * scale)))
        norm = normalize_scanned_image(resized)
        phash = compute_image_phash(norm)
        dist = phash - self.base_phash
        assert dist == 0, f"scale={scale}: Hamming={dist}, expected 0"


class TestRotationTolerance:
    """Document rotation after deskew: measure remaining Hamming distance.

    The deskew step corrects most rotation, but synthetic images may lack
    strong text baselines for Hough detection. Real scanned documents
    typically perform better.
    """

    def setup_method(self):
        self.base = generate_synthetic_document()
        self.base_norm = normalize_scanned_image(self.base)
        self.base_phash = compute_image_phash(self.base_norm)

    @pytest.mark.parametrize("angle", [2, -2])
    def test_small_rotation_bounded(self, angle):
        """±2° rotation should produce bounded Hamming distance."""
        rotated = rotate_image(self.base, angle)
        norm = normalize_scanned_image(rotated)
        phash = compute_image_phash(norm)
        dist = phash - self.base_phash
        assert dist <= 15, f"angle={angle}: Hamming={dist} > 15 (unexpectedly high)"


class TestClusteringEndToEnd:
    """Full pipeline: generate images -> normalize -> phash -> cluster."""

    def test_shifted_variants_cluster_together(self):
        """Images shifted by ±20px should cluster together at threshold=5."""
        base = generate_synthetic_document()

        with tempfile.TemporaryDirectory() as tmpdir:
            cv2.imwrite(os.path.join(tmpdir, "original.png"), base)
            cv2.imwrite(os.path.join(tmpdir, "shift_h+20.png"), shift_image(base, 20))
            cv2.imwrite(os.path.join(tmpdir, "shift_h-20.png"), shift_image(base, -20))
            cv2.imwrite(os.path.join(tmpdir, "shift_v+20.png"), shift_image(base, dy=20))

            clusters = find_duplicate_clusters(tmpdir, hamming_threshold=5, normalize=True)

            multi = [c for c in clusters if c["cluster_size"] > 1]
            assert len(multi) >= 3, (
                f"Expected at least 3 images in one cluster, got sizes: "
                f"{[c['cluster_size'] for c in clusters]}"
            )

    def test_different_document_not_clustered(self):
        """A genuinely different document should NOT join the main cluster."""
        doc_a = generate_synthetic_document(width=800, height=1100)
        doc_b = generate_synthetic_document(width=600, height=800)

        with tempfile.TemporaryDirectory() as tmpdir:
            cv2.imwrite(os.path.join(tmpdir, "doc_a.png"), doc_a)
            cv2.imwrite(os.path.join(tmpdir, "doc_b.png"), doc_b)

            clusters = find_duplicate_clusters(tmpdir, hamming_threshold=5, normalize=True)

            for r in clusters:
                assert r["cluster_size"] == 1, (
                    f"Different documents should not cluster: {r['original_path']}"
                )

    def test_csv_report_format(self):
        """CSV report should have correct columns."""
        base = generate_synthetic_document()
        with tempfile.TemporaryDirectory() as tmpdir:
            cv2.imwrite(os.path.join(tmpdir, "doc.png"), base)
            report_path = os.path.join(tmpdir, "report.csv")

            clusters = find_duplicate_clusters(tmpdir, hamming_threshold=5)
            path = export_dedup_report(clusters, report_path)

            assert os.path.exists(path)
            with open(path) as f:
                header = f.readline().strip()
            assert "original_path" in header
            assert "cluster_id" in header
            assert "hamming_distance_from_representative" in header
            assert "cluster_size" in header
