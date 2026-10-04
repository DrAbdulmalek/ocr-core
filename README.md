# ocr-core

مستودع OCR المشترك — محرك قواعد "ميثاق الأدلة البصرية" (18 قاعدة) + تصحيحات طبية عربية + محركات OCR + معالجة مسبقة/لاحقة + معايير قياس.

## الحالة
- [x] هيكل المستودع (stage 2)
- [x] نقل محرك القواعد (stage 3)
- [x] التصحيحات العربية (stage 4)
- [x] محرك Tesseract (stage 5)
- [x] محرك PaddleOCR منقول من omni-medical-suite — extra: `pip install .[paddle]`
- [x] محرك EasyOCR منقول من omni-medical-suite — extra: `pip install .[easyocr]`
- [x] EnsembleEngine + EngineRouter + telemetry (PR #6)
- [x] Preprocess (scanner_fixer) (PR #6) — extra: `pip install .[preprocess]`
- [x] Postprocess طبي: normalization / field_extractor / deduplication (PR #6)
- [x] Benchmarks: suite كاملة + ci_suite (بوابات CER/WER، A/B) (PR #6)
- [x] التكامل مع marathon_ted_pipeline (PR #14 مدموج — shim + pin)
- [ ] التكامل الكامل مع omni-medical-suite (استهلاك ocr-core بدل المكررات)

## الوحدات

| المسار | الوظيفة |
|---|---|
| `ocr_core.rules` | محرك قواعد الميثاق — الافتراضي المدمج: `ocr_core/config/marathon_ocr_rules.yaml` (ميثاق v2) مع pre-pass شقيق `text_normalization_rules.yaml` (18 قاعدة) |
| `ocr_core.engines` | base (OCREngine/OCRResult) + tesseract + paddle + easyocr + ensemble + router |
| `ocr_core.preprocess` | crop/deskew/rotate/enhance/normalize/dedup/text_dedup/pipeline |
| `ocr_core.postprocess` | corrections_ar (311 زوجًا) + normalization + field_extractor + deduplication + medical_terms.json |
| `ocr_core.benchmarks` | suite القياس الكاملة + ci_suite (threshold_checker + ab_testing) |
| `ocr_core.rtl_utils` | معالجة العربية RTL (خرائط مبنية وقت الاستيراد) |
| `ocr_core.telemetry` | سجل قرارات مهيكل (بديل decision_log) |

**سياسة الثقة:** الثقة `0.0` = مجهولة — لا تُختلق أبدًا (`engines/base.py`). المحركات تبلّغ الأخطاء عبر `OCRResult.error` ولا ترفع استثناءات.

## التثبيت
```bash
pip install -e ".[dev,preprocess,benchmarks]"   # لتشغيل مجموعة الاختبارات كاملة (كما في CI)
# optional engines / modules
pip install -e ".[tesseract]"   # pytesseract
pip install -e ".[paddle]"      # PaddleOCR
pip install -e ".[easyocr]"     # EasyOCR
pip install -e ".[preprocess]"  # opencv-headless + imagehash + rapidfuzz
pip install -e ".[benchmarks]"  # rapidfuzz + click + rich
pip install -e ".[all]"         # الكل
```

## الاستخدام
```python
from ocr_core.rules.engine import OCRProcessor
p = OCRProcessor()  # الافتراضي: ocr_core/config/marathon_ocr_rules.yaml (ميثاق v2)
# التجاوز: OCRProcessor(rules_file=...) أو المتغير البيئي MARATHON_OCR_RULES

from ocr_core.engines.paddle import PaddleEngine
r = PaddleEngine().process_image("scan.png")  # r.error عند غياب paddleocr

from ocr_core.engines.tesseract import TesseractEngine
from ocr_core.engines.ensemble import EnsembleEngine
ens = EnsembleEngine({"tesseract": TesseractEngine(), "paddle": PaddleEngine()})
best = ens.process_image("scan.png")  # أفضل نتيجة: ثقة×طول×صلاحية

from ocr_core.preprocess import fix_scan
fixed = fix_scan("scan.png")  # crop → deskew → rotate → denoise → enhance

from ocr_core.postprocess.field_extractor import ArabicMedicalFieldExtractor
fields = ArabicMedicalFieldExtractor().extract_fields(fixed_text)
```

## المصدر والترخيص
معظم الوحدات منقولة من [omni-medical-suite](https://github.com/DrAbdulmalek/omni-medical-suite) (MIT) مع انحرافات موثقة داخل docstrings كل وحدة — التفاصيل في PR #6.
