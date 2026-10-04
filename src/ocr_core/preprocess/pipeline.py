"""
pipeline.py
Main entry point: orchestrates crop → rotate → deskew → enhance
Returns processed image and a metadata report.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

from .crop import auto_crop, auto_crop_contour
from .deskew import deskew
from .rotate import auto_rotate
from .enhance import enhance_for_ocr, get_estimated_dpi


def fix_scan(
    input_path: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    # Pipeline control
    do_crop: bool = True,
    do_rotate: bool = False,
    do_deskew: bool = True,
    do_enhance: bool = True,
    # Enhancement options
    binarize: bool = False,
    target_dpi: Optional[int] = 300,
    use_tesseract_osd: bool = False,
    deskew_method: str = "hough",
    # Crop options
    crop_padding: int = 10,
) -> Dict[str, Any]:
    """
    Full scanner image correction pipeline.

    Order of operations:
    1. Load image
    2. Auto-crop borders
    3. Detect and correct 180° rotation
    4. Deskew (fix small tilt)
    5. Enhance for OCR

    Args:
        input_path: Path to image file, or numpy array (BGR)
        output_path: Where to save result. If None, result returned only in dict.
        do_crop: Enable border cropping
        do_rotate: Enable 180° rotation detection (default: False — heuristic is
                   unreliable; enable explicitly or use_tesseract_osd=True)
        do_deskew: Enable skew correction
        do_enhance: Enable OCR enhancement
        binarize: Convert to B&W (good for text-only pages)
        target_dpi: Target DPI for OCR (300 recommended)
        use_tesseract_osd: Use Tesseract for rotation detection (more accurate)
        deskew_method: "hough" or "projection"
        crop_padding: Padding in pixels around detected content

    Returns:
        dict with keys:
            - image: Final processed numpy array (BGR)
            - steps: Dict of intermediate images {step_name: array}
            - report: Processing report with angles, detected DPI, etc.
    """
    report = {}
    steps = {}

    # ─── Load ───────────────────────────────────────────────────────────────
    if isinstance(input_path, np.ndarray):
        image = input_path.copy()
        report["source"] = "array"
    else:
        path = Path(input_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        report["source"] = str(path)
        report["original_size"] = f"{image.shape[1]}x{image.shape[0]}"

    steps["original"] = image.copy()

    # Estimate source DPI
    estimated_dpi = get_estimated_dpi(image)
    report["estimated_dpi"] = estimated_dpi

    # ─── Step 1: Crop ────────────────────────────────────────────────────────
    if do_crop:
        cropped = auto_crop_contour(image, padding=crop_padding)
        # Fall back if contour crop removed too much
        if _is_reasonable_crop(image, cropped):
            image = cropped
        else:
            image = auto_crop(image, padding=crop_padding)
        steps["cropped"] = image.copy()
        report["crop"] = f"{image.shape[1]}x{image.shape[0]}"

    # ─── Step 2: Auto-rotate (180°) ──────────────────────────────────────────
    if do_rotate:
        image, rotation_applied = auto_rotate(image, use_tesseract=use_tesseract_osd)
        steps["rotated"] = image.copy()
        report["rotation_applied_deg"] = rotation_applied

    # ─── Step 3: Deskew ──────────────────────────────────────────────────────
    if do_deskew:
        image, skew_angle, deskew_meta = deskew(image, method=deskew_method)
        steps["deskewed"] = image.copy()
        report["skew_corrected_deg"] = round(skew_angle, 2)
        report["deskew_meta"] = deskew_meta

    # ─── Step 4: Enhance ────────────────────────────────────────────────────
    if do_enhance:
        image = enhance_for_ocr(
            image,
            target_dpi=target_dpi if estimated_dpi else None,
            source_dpi=estimated_dpi,
            binarize=binarize
        )
        steps["enhanced"] = image.copy()

    report["final_size"] = f"{image.shape[1]}x{image.shape[0]}"
    report["is_color"] = len(image.shape) == 3

    # ─── Save ────────────────────────────────────────────────────────────────
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), image)
        report["saved_to"] = str(out)

    return {
        "image": image,
        "steps": steps,
        "report": report,
    }


def fix_scan_batch(
    input_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    suffix: str = "_fixed",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Process multiple scanned images.

    Args:
        input_paths: List of image file paths
        output_dir: Directory for processed images
        suffix: Appended to filename before extension (e.g. "scan_fixed.png")
        **kwargs: Passed to fix_scan()

    Returns:
        List of result dicts (one per image)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for path in input_paths:
        path = Path(path)
        out_name = f"{path.stem}{suffix}{path.suffix}"
        out_path = output_dir / out_name

        try:
            result = fix_scan(path, output_path=out_path, **kwargs)
            result["input"] = str(path)
            result["status"] = "ok"
        except Exception as e:
            result = {
                "input": str(path),
                "status": "error",
                "error": str(e),
            }

        results.append(result)
        print(f"[{'OK' if result['status'] == 'ok' else 'ERROR'}] {path.name}")

    return results


def _is_reasonable_crop(original: np.ndarray, cropped: np.ndarray) -> bool:
    """
    Sanity check: cropped result should keep at least 30% of original area.
    """
    orig_area = original.shape[0] * original.shape[1]
    crop_area = cropped.shape[0] * cropped.shape[1]
    return crop_area >= orig_area * 0.30
