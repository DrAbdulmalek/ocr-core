# ocr-core

مستودع OCR المشترك — محرك قواعد "ميثاق الأدلة البصرية" (18 قاعدة) + تصحيحات طبية عربية.

## الحالة
- [x] هيكل المستودع (stage 2)
- [x] نقل محرك القواعد (stage 3)
- [x] التصحيحات العربية (stage 4)
- [x] محرك Tesseract (stage 5)
- [x] محرك PaddleOCR منقول من omni-medical-suite (`src/ocr/paddle_engine.py`) — extra: `pip install .[paddle]`
- [ ] EasyOCR / ensemble / scanner_fixer من omni (قادم)
- [ ] التكامل مع marathon_ted_pipeline (stage 6)

## التثبيت
```bash
pip install -e ".[dev]"
# optional engines
pip install -e ".[tesseract]"
pip install -e ".[paddle]"
```

## الاستخدام
```python
from ocr_core.rules.engine import OCRProcessor
p = OCRProcessor()  # يستخدم charter_v2.yaml المدمج

from ocr_core.engines.paddle import PaddleEngine
r = PaddleEngine().process_image("scan.png")  # error field if paddleocr missing
```
