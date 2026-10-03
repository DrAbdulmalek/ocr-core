"""تصحيحات OCR عربية — من dataset DrAbdulmalek/arabic-medical-ocr-corrections."""
import json
from pathlib import Path
from typing import Optional

_DEFAULT = Path(__file__).parent / "data" / "corrections_ar.jsonl"

class ArabicMedicalCorrections:
    def __init__(self, data_path: Optional[Path] = None):
        self.path = data_path or _DEFAULT
        if not self.path.exists():
            raise FileNotFoundError(f"corrections not found: {self.path}")
        self.corrections: dict[str, str] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                wrong = rec.get("incorrect_ocr_output") or rec.get("wrong")
                right = rec.get("correct_text") or rec.get("right")
                if wrong and right:
                    self.corrections[wrong] = right
            except json.JSONDecodeError:
                continue

    def apply(self, text: str) -> tuple[str, int]:
        count = 0
        for wrong, right in self.corrections.items():
            if wrong in text:
                text = text.replace(wrong, right)
                count += 1
        return text, count
