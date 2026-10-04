"""
text_dedup.py
=============

Two-stage document deduplication pipeline:

  Stage 1 — cheap phash filter (generous threshold) → candidate pairs
  Stage 2 — OCR text extraction + fuzzy similarity → confirmed duplicates

This module never deletes any image.  It only produces a structured
report for human review.

Design rationale (see REAL_DATA_VALIDATION.md v2):
  phash alone is insufficient for real scanned documents because
  auto_crop produces different bounding boxes for same-content scans,
  causing high Hamming distances (up to 22 for genuinely identical
  documents).  OCR text comparison is inherently invariant to the
  visual problems that plague phash (different margins, slight
  rotation, sensor noise).
"""

from __future__ import annotations

import csv
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import imagehash
import numpy as np
from PIL import Image
from rapidfuzz import fuzz

from ocr_core.preprocess.dedup import compute_image_phash, _normalize_image
from ocr_core.preprocess.normalize import normalize_scanned_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------

def _clean_ocr_text(text: str) -> str:
    """Normalise OCR output for robust comparison.

    * Collapse all whitespace (spaces, tabs, newlines) into single spaces
    * Strip leading/trailing whitespace
    * Remove isolated punctuation that OCR sometimes hallucinates
    """
    if not text:
        return ""
    # Replace all whitespace runs with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove isolated dots, dashes, underscores that are not part of numbers
    text = re.sub(r"(?<!\d)[._-](?!\d)", "", text)
    return text.strip()


def extract_text_for_comparison(
    image: np.ndarray,
    ocr_func: Callable,
    languages: Optional[list[str]] = None,
) -> str:
    """Run OCR on an image and return cleaned text for comparison.

    The image is normalised (deskew + auto_crop, default fit_mode) before
    OCR to maximise recognition quality.  The fit_mode does not matter for
    text extraction — it only affects pixel geometry, not textual content.

    Args:
        image: BGR or grayscale numpy array.
        ocr_func: Callable that accepts a numpy array and returns a string.
            Can be ``lambda img: pytesseract.image_to_string(img, lang='ara+eng')``
            or ``lambda img: engine.recognize(img)['text']`` for the full
            ``OCREngine`` wrapper.
        languages: Optional hint for OCR language selection (logged only,
            the ocr_func is responsible for handling it).

    Returns:
        Cleaned OCR text string.
    """
    # Normalise for OCR quality (not for hashing)
    normalised = normalize_scanned_image(image)

    if languages:
        logger.debug("OCR languages: %s", languages)

    raw_text = ocr_func(normalised)
    if isinstance(raw_text, dict):
        # OCREngine.recognize() returns a dict with 'text' key
        raw_text = raw_text.get("text", "")
    elif not isinstance(raw_text, str):
        raw_text = str(raw_text)

    return _clean_ocr_text(raw_text)


def fuzzy_text_similarity(text1: str, text2: str) -> float:
    """Compute fuzzy similarity ratio between two text strings.

    Uses ``rapidfuzz.fuzz.ratio`` which implements normalised Levenshtein
    distance (0–100 scale).  This is the same library already used
    elsewhere in the project (medical_ocr_postprocessor, benchmarks).

    Args:
        text1: First text (cleaned OCR output).
        text2: Second text (cleaned OCR output).

    Returns:
        Similarity score between 0.0 and 100.0.
    """
    if not text1 and not text2:
        return 100.0
    if not text1 or not text2:
        return 0.0

    return fuzz.ratio(text1, text2)


def confirm_duplicate(
    image1: np.ndarray,
    image2: np.ndarray,
    ocr_func: Callable,
    similarity_threshold: float = 85.0,
    languages: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Two-stage duplicate confirmation: phash pre-check + OCR text match.

    Args:
        image1: First image (BGR numpy array).
        image2: Second image (BGR numpy array).
        ocr_func: OCR callable (see :func:`extract_text_for_comparison`).
        similarity_threshold: Minimum text similarity (0–100) to confirm
            a duplicate.  Default 85.0.
        languages: Optional language hint for OCR.

    Returns:
        Dict with keys:
            - duplicate (bool): confirmed duplicate?
            - phash_distance (int): Hamming distance (informational)
            - text_similarity (float): 0–100 fuzzy ratio
            - text1 (str): extracted text from image1
            - text2 (str): extracted text from image2
    """
    # Compute phash distance (informational — not used for gating here,
    # the caller already used phash for candidate selection)
    h1 = compute_image_phash(_normalize_image(image1))
    h2 = compute_image_phash(_normalize_image(image2))
    phash_distance = h1 - h2

    # Extract text from both images
    text1 = extract_text_for_comparison(image1, ocr_func, languages)
    text2 = extract_text_for_comparison(image2, ocr_func, languages)

    # Compute fuzzy similarity
    similarity = fuzzy_text_similarity(text1, text2)

    is_duplicate = similarity >= similarity_threshold

    logger.debug(
        "confirm_duplicate: phash=%d, text_sim=%.1f%%, threshold=%.1f%% → %s",
        phash_distance,
        similarity,
        similarity_threshold,
        "DUPLICATE" if is_duplicate else "NOT duplicate",
    )

    return {
        "duplicate": is_duplicate,
        "phash_distance": phash_distance,
        "text_similarity": round(similarity, 2),
        "text1": text1,
        "text2": text2,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def find_true_duplicates(
    image_paths: list[str],
    ocr_func: Callable,
    phash_threshold: int = 25,
    text_similarity_threshold: float = 85.0,
    languages: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Two-stage dedup pipeline: phash candidate selection + OCR confirmation.

    Stage 1: Group images by phash similarity (generous threshold).
    Stage 2: For each candidate pair within a cluster, confirm via OCR
    text fuzzy matching.

    Args:
        image_paths: List of file paths to scanned images.
        ocr_func: OCR callable (see :func:`extract_text_for_comparison`).
        phash_threshold: Generous Hamming threshold for candidate selection.
            Default 25 (out of 64) — intentionally permissive to avoid
            missing candidates.  False positives at this stage are filtered
            out by OCR in Stage 2.
        text_similarity_threshold: Minimum text similarity (0–100) to
            confirm a duplicate.  Default 85.0.
        languages: Optional language hint for OCR.

    Returns:
        List of result dicts, one per compared pair, with keys:
            - image1, image2: file paths
            - phash_distance: Hamming distance (Stage 1)
            - text_similarity: 0–100 fuzzy ratio (Stage 2)
            - confirmed_duplicate: bool
            - text1, text2: extracted texts (for manual review)
    """
    # ── Stage 1: Load images and compute phashes ──
    logger.info(
        "Stage 1: Computing phashes for %d images (threshold=%d)",
        len(image_paths),
        phash_threshold,
    )

    images: dict[str, np.ndarray] = {}
    hashes: dict[str, imagehash.ImageHash] = {}

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("Could not read: %s", path)
            continue
        images[path] = img
        hashes[path] = compute_image_phash(_normalize_image(img))

    # ── Stage 1: Greedy clustering ──
    clusters: list[tuple[str, list[str]]] = []  # [(representative_path, [member_paths])]
    path_to_cluster: dict[str, int] = {}  # path → cluster index

    for path in hashes:
        best_dist = 64
        best_idx = None
        for idx, (rep_path, _members) in enumerate(clusters):
            dist = hashes[path] - hashes[rep_path]
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is not None and best_dist <= phash_threshold:
            clusters[best_idx][1].append(path)
            path_to_cluster[path] = best_idx
        else:
            clusters.append((path, [path]))
            path_to_cluster[path] = len(clusters) - 1

    multi_clusters = [(rep, members) for rep, members in clusters if len(members) > 1]
    logger.info(
        "Stage 1: %d multi-image clusters from %d images",
        len(multi_clusters),
        len(images),
    )

    # ── Stage 2: OCR confirmation for each pair in multi-image clusters ──
    logger.info("Stage 2: OCR confirmation for candidate pairs")

    results: list[dict[str, Any]] = []

    for rep_path, members in multi_clusters:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                p1, p2 = members[i], members[j]
                if p1 not in images or p2 not in images:
                    continue

                start = time.time()
                confirmation = confirm_duplicate(
                    images[p1],
                    images[p2],
                    ocr_func=ocr_func,
                    similarity_threshold=text_similarity_threshold,
                    languages=languages,
                )
                elapsed = time.time() - start

                results.append({
                    "image1": p1,
                    "image2": p2,
                    "phash_distance": confirmation["phash_distance"],
                    "text_similarity": confirmation["text_similarity"],
                    "confirmed_duplicate": confirmation["duplicate"],
                    "text1": confirmation["text1"],
                    "text2": confirmation["text2"],
                    "ocr_time_sec": round(elapsed, 2),
                })

                status = "✅ DUPLICATE" if confirmation["duplicate"] else "❌ NOT duplicate"
                logger.info(
                    "  %s vs %s: phash=%d, text_sim=%.1f%% → %s (%.1fs)",
                    Path(p1).name,
                    Path(p2).name,
                    confirmation["phash_distance"],
                    confirmation["text_similarity"],
                    status,
                    elapsed,
                )

    # Summary
    confirmed = sum(1 for r in results if r["confirmed_duplicate"])
    rejected = len(results) - confirmed
    logger.info(
        "Stage 2 complete: %d/%d pairs confirmed as duplicates, %d rejected",
        confirmed,
        len(results),
        rejected,
    )

    return results


def export_text_dedup_report(
    results: list[dict[str, Any]],
    output_csv: str,
) -> str:
    """Export two-stage dedup results to CSV.

    Args:
        results: Output from :func:`find_true_duplicates`.
        output_csv: Path to output CSV file.

    Returns:
        Absolute path to the written CSV file.
    """
    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image1",
        "image2",
        "phash_distance",
        "text_similarity",
        "confirmed_duplicate",
        "text1",
        "text2",
        "ocr_time_sec",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Text dedup report written to %s (%d rows)", output_path, len(results))
    return str(output_path)