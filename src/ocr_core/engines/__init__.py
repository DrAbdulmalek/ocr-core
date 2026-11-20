"""Engines subpackage — optional engine dependencies stay lazy."""
from ocr_core.engines.base import OCRResult, OCREngine
from ocr_core.engines.abbyy import AbbyyFineReaderEngine

__all__ = ["OCRResult", "OCREngine", "AbbyyFineReaderEngine"]
