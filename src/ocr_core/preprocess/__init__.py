"""Pre-OCR image normalization for scanned documents.

Ported from omni-medical-suite / packages/scanner_fixer (v1.2.0), MIT.
Pipeline: crop → deskew → rotate → denoise → enhance contrast.
Orchestration layers (cli, batch_pipeline, pdf_ocr_processor) were
intentionally NOT ported — they are tied to the omni application context.
"""
from .pipeline import fix_scan, fix_scan_batch
from .crop import auto_crop
from .deskew import deskew
from .rotate import auto_rotate
from .enhance import enhance_for_ocr
from .enhanced_preprocessor import DocumentPreprocessor as EnhancedPreprocessor
from .binarize import (
    to_grayscale,
    otsu_threshold,
    binarize_otsu,
    binarize_sauvola,
    illumination_uniformity,
    binarize_auto,
)

__version__ = "1.2.0"
__all__ = [
    "fix_scan",
    "fix_scan_batch",
    "auto_crop",
    "deskew",
    "auto_rotate",
    "enhance_for_ocr",
    "EnhancedPreprocessor",
    # binarization suite (docs/09 §2.2 — Otsu/Sauvola/auto router)
    "to_grayscale",
    "otsu_threshold",
    "binarize_otsu",
    "binarize_sauvola",
    "illumination_uniformity",
    "binarize_auto",
]
