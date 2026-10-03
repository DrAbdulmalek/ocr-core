# ocr-core

مستودع OCR المشترك — محرك قواعد "ميثاق الأدلة البصرية" (18 قاعدة) + تصحيحات طبية عربية.

## الحالة
- [x] هيكل المستودع (stage 2)
- [ ] نقل محرك القواعد (stage 3)
- [ ] التصحيحات العربية (stage 4)
- [ ] محرك Tesseract (stage 5)
- [ ] التكامل مع marathon_ted_pipeline (stage 6)

## التثبيت
```bash
pip install -e ".[dev]"
```

## الاستخدام
```python
from ocr_core.rules.engine import OCRProcessor
p = OCRProcessor()  # يستخدم charter_v2.yaml المدمج
```
