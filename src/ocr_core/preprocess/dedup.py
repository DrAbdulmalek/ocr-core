#!/usr/bin/env python3
"""
scanner_fixer.dedup
==================

Standalone dedup script for scanned medical document batches.

Scans a folder of images, computes perceptual hashes (phash), and groups
images into duplication clusters using Hamming distance. Outputs a CSV
report but **never deletes any image automatically**.

Usage:
    # Basic usage
    python -m scanner_fixer.dedup /path/to/scans --output dedup_report.csv

    # Custom threshold (more aggressive dedup)
    python -m scanner_fixer.dedup /path/to/scans --threshold 8 --output report.csv

    # Skip normalization
    python -m scanner_fixer.dedup /path/to/scans --no-normalize --output report.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Supported image extensions for scanning
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def compute_image_phash(
    image: np.ndarray, hash_size: int = 8
) -> imagehash.ImageHash:
    """Compute perceptual hash for a normalized image.

    Args:
        image: Grayscale or BGR numpy array.
        hash_size: Size of the phash grid (default 8 → 64-bit hash).

    Returns:
        imagehash.ImageHash object.
    """
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image.copy()
    )
    pil_image = Image.fromarray(gray)
    return imagehash.phash(pil_image, hash_size=hash_size)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Apply scanner normalization pipeline to an image.

    Falls back to basic contrast normalization if the full pipeline
    is unavailable (the normalize module may be created in a parallel task).
    """
    try:
        from ocr_core.preprocess.normalize import normalize_scanned_image

        return normalize_scanned_image(image)
    except ImportError:
        logger.debug(
            "ocr_core.preprocess.normalize not available — using basic normalization"
        )
        # Basic fallback: resize, normalize contrast, slight blur
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        return blurred


def find_duplicate_clusters(
    image_folder: str,
    hamming_threshold: int = 5,
    hash_size: int = 8,
    normalize: bool = True,
) -> list[dict[str, Any]]:
    """Scan a folder and group duplicate images by perceptual hash similarity.

    Uses a greedy clustering approach: each image is compared against existing
    cluster representatives. If the Hamming distance to the representative is
    within the threshold, the image joins that cluster. Otherwise, a new
    cluster is created.

    Args:
        image_folder: Path to directory containing image files.
        hamming_threshold: Maximum Hamming distance to consider two images
            duplicates (out of hash_size^2 bits). Default 5 out of 64.
        hash_size: phash grid size. Default 8 → 64-bit hash.
        normalize: Whether to apply image normalization before hashing.

    Returns:
        List of dicts with keys:
            - original_path: absolute path to the image
            - cluster_id: identifier like "cluster_001" or "unique_001"
            - hamming_distance_from_representative: distance to cluster rep
            - cluster_size: total images in the cluster
    """
    folder_path = Path(image_folder).resolve()
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {image_folder}")

    # Collect image files
    image_files = sorted(
        p
        for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        logger.warning(f"No image files found in {image_folder}")
        return []

    logger.info(
        f"Scanning {len(image_files)} images in {image_folder} "
        f"(threshold={hamming_threshold}, hash_size={hash_size})"
    )

    # Cluster data: list of (representative_hash, cluster_id)
    clusters: list[tuple[imagehash.ImageHash, str]] = []
    # Results accumulation
    results: list[dict[str, Any]] = []
    # Track cluster sizes
    cluster_sizes: dict[str, int] = {}
    cluster_counter = 0
    unique_counter = 0

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Could not read image: {img_path}")
            continue

        if normalize:
            image = _normalize_image(image)

        try:
            phash = compute_image_phash(image, hash_size=hash_size)
        except Exception as e:
            logger.warning(f"Failed to compute phash for {img_path}: {e}")
            continue

        # Find closest cluster representative
        best_distance = hash_size * hash_size  # max possible distance
        best_cluster_id = None

        for rep_hash, cid in clusters:
            dist = phash - rep_hash  # Hamming distance (int)
            if dist < best_distance:
                best_distance = dist
                best_cluster_id = cid

        if best_cluster_id is not None and best_distance <= hamming_threshold:
            # Join existing cluster
            cid = best_cluster_id
            cluster_sizes[cid] += 1
        else:
            # Create new cluster
            cluster_counter += 1
            cid = f"cluster_{cluster_counter:03d}"
            clusters.append((phash, cid))
            cluster_sizes[cid] = 1

        results.append({
            "original_path": str(img_path),
            "cluster_id": cid,
            "hamming_distance_from_representative": 0 if best_cluster_id is None else best_distance,
            "cluster_size": cluster_sizes[cid],
        })

    # Rename single-image clusters to "unique_XXX" for clarity
    # First pass: identify single-image clusters
    single_image_clusters = {
        cid for cid, size in cluster_sizes.items() if size == 1
    }

    # Second pass: rename them
    for r in results:
        if r["cluster_id"] in single_image_clusters:
            unique_counter += 1
            r["cluster_id"] = f"unique_{unique_counter:03d}"

    # Summary
    dup_clusters = sum(1 for s in cluster_sizes.values() if s > 1)
    dup_images = sum(s for s in cluster_sizes.values() if s > 1)
    logger.info(
        f"Found {dup_clusters} duplication clusters ({dup_images} images) "
        f"and {len(single_image_clusters)} unique images"
    )

    return results


def export_dedup_report(
    clusters: list[dict[str, Any]],
    output_csv: str,
) -> str:
    """Export dedup report to CSV.

    Args:
        clusters: Result from find_duplicate_clusters.
        output_csv: Path to output CSV file.

    Returns:
        Absolute path to the written CSV file.
    """
    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "original_path",
        "cluster_id",
        "hamming_distance_from_representative",
        "cluster_size",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clusters)

    logger.info(f"Dedup report written to {output_path} ({len(clusters)} rows)")
    return str(output_path)


def main() -> None:
    """CLI entry point for batch dedup detection."""
    parser = argparse.ArgumentParser(
        description=(
            "Scan a folder of images for duplicates using perceptual hashing. "
            "Outputs a CSV report — does NOT delete any files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s /path/to/scans --output dedup_report.csv
  %(prog)s /path/to/scans --threshold 8 --output report.csv
  %(prog)s /path/to/scans --no-normalize --output report.csv
        """,
    )
    parser.add_argument(
        "image_folder",
        help="Path to folder containing scanned images",
    )
    parser.add_argument(
        "--output", "-o",
        default="dedup_report.csv",
        help="Output CSV path (default: dedup_report.csv)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=5,
        help=(
            "Hamming distance threshold for duplicates "
            "(out of 64 bits, default: 5)"
        ),
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="phash grid size (default: 8 → 64-bit hash)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip image normalization before hashing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        clusters = find_duplicate_clusters(
            image_folder=args.image_folder,
            hamming_threshold=args.threshold,
            hash_size=args.hash_size,
            normalize=not args.no_normalize,
        )

        if clusters:
            report_path = export_dedup_report(clusters, args.output)

            # Print summary
            multi = [c for c in clusters if c["cluster_size"] > 1]
            unique = [c for c in clusters if c["cluster_size"] == 1]
            print(f"\nDedup Report: {report_path}")
            print(f"  Total images:    {len(clusters)}")
            print(f"  Unique images:   {len(unique)}")
            print(f"  In clusters:     {len(multi)}")
            if multi:
                dup_cluster_ids = set(c["cluster_id"] for c in multi)
                print(f"  Dup clusters:    {len(dup_cluster_ids)}")
                print(
                    f"\n  ⚠ Review clusters in the CSV before manually removing duplicates."
                )
        else:
            print("No images found or all images are unique.")
            sys.exit(0)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
