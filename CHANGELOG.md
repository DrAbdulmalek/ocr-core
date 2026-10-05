# سجل التغييرات — marathon-ocr-core

> ملاحظة اسم الحزمة: الاسم الحقيقي للتوزيع هو **`marathon-ocr-core`**.
> من يثبّت من git عليه استخدام هذا الاسم (وليس `ocr-core`) وإلا فشل حلّ pip:
>
> ```
> marathon-ocr-core[tesseract] @ git+https://github.com/DrAbdulmalek/ocr-core.git@v0.5.0
> ```

## v0.5.0 (قادم — إصدار عقد الاستهلاك المركزي)

هذا الإصدار هو المرجع الذي تعقد عليه المستودعات المستهلكة (intelli-file-manager وغيرها)
بصيغة "v0.5.0+ مع الواجهة العامة" في مستندات harness.

### أُضيف
- شيم مستودعي `ocr_core/` في الجذر (نسخة + ميثاق v2 + اختبار سلامة) للاستيراد من
  جذر المستودع دون تثبيت؛ الكود الفعلي يبقى في الحزمة المثبتة `src/ocr_core`.
- اختبارات: `test_charter_integrity.py` (سلامة شيم الجذر) و`test_charter_sync.py`
  (تثبيت ميثاق v2 المدمج وقواعد الـ pre-pass ضد الانجراف).

### تغيّر
- **مصدر واحد للميثاق**: حذف `src/ocr_core/rules/data/charter_v2.yaml` اليتيم
  (تعارض مع النسخة المدمجة)؛ الافتراضي المدمج الموثق:
  `ocr_core/config/marathon_ocr_rules.yaml` (ميثاق v2 — 18 قاعدة) مع متغير
  البيئة `MARATHON_OCR_RULES` للتحايل.
- **بيانات إصدار حقيقية**: `ocr_core.__version__` يُقرأ من `importlib.metadata`
  (توزيع `marathon-ocr-core`) بدل ثابت `0.1.0` القديم؛ و`0.0.0.dev0` عند
  checkout المصدر دون تثبيت.
- إصدار الحزمة: `0.4.0` → `0.5.0` (بعد 0.4.1 الوسيطي على فرع PR#7).

### المحتوى الحي (مستقر من v0.4.0)
- `ocr_core.rules` — محرك قواعد الميثاق.
- `ocr_core.engines` — base + tesseract + paddle + easyocr + ensemble + router.
- `ocr_core.preprocess` — crop/deskew/rotate/enhance/normalize/dedup.
- `ocr_core.postprocess` — تصحيحات عربية طبية (311 زوجًا) + normalization +
  field_extractor + deduplication.
- `ocr_core.benchmarks` — suite كاملة + ci_suite (بوابات CER/WER، A/B).
- `ocr_core.rtl_utils` + `ocr_core.telemetry`.
- extras: `tesseract` / `paddle` / `easyocr` / `surya` / `preprocess` / `benchmarks` / `all` / `dev`.

### مؤجل لهذا الإصدار
- PR #11 (محرك Surya — draft) وPR #12 (نماذج tessdata_fast ara+eng — draft):
  لم تُدمج بعد؛ ستأتي في إصدار لاحق (v0.6.0) وفق قرار المالك.

## v0.4.0
- إصدار مرحلي: ensemble + router + telemetry + preprocess (scanner_fixer) +
  postprocess طبي + benchmarks كاملة (PR #6)، ونقل محرك القواعد والتصحيحات
  العربية من omni-medical-suite.
