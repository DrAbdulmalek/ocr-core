"""استيراد كل builtins لتسجيلها في السجل العام."""
from . import io
from . import preprocess
from . import ocr
from . import postprocess
from . import export
from . import benchmark
from . import detect
from . import vlm_ocr              # ← جديد

__all__ = [
    "io", "preprocess", "ocr", "postprocess",
    "export", "benchmark", "detect", "vlm_ocr",
]
