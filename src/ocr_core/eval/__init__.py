"""ocr_core.eval — تقييم دقة OCR: تطبيع مُعلن الإصدار + مقاييس CER/WER.

النقطة القانونية الوحيدة للقياس في النواة (F-13/F-16):
    from ocr_core.eval import calculate_cer, calculate_wer, normalize_v1
"""
from ocr_core.eval.metrics import calculate_cer, calculate_wer
from ocr_core.eval.normalize import (
    NORMALIZE_V1_STEPS,
    NORMALIZE_V1_VERSION,
    OPTIONAL_STEPS,
    normalize_v1,
)

__all__ = [
    "NORMALIZE_V1_STEPS",
    "NORMALIZE_V1_VERSION",
    "OPTIONAL_STEPS",
    "calculate_cer",
    "calculate_wer",
    "normalize_v1",
]
