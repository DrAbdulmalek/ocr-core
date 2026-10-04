
## 2026-10-05 — محرك Surya (فرع feat/add-surya-engine)

- live-verified surya-ocr 0.22.1 API قبل الكتابة (RecognitionPredictor بلا langs — multilingual تلقائي؛ PageOCRResult.blocks + html + reading_order؛ DetectionPredictor client-backed افتراضيًا). الكود القديم من وثائق <0.17 كان سينكسر.
- engines/surya_engine.py: سياسة base.py (الخطأ→error، الثقة 0.0=غير معروف أبدًا لا مختلقة)، PDF عبر pypdfium2 (ليس pymupdf AGPL)، html→text أصلي stdlib.
- pyproject: extra surya مرفوع للحد الأدنى 0.22 (كان 0.6 — أسماء API غير متوافقة).
- الاختبارات: surya 6 passed/1 skip؛ كامل الحزمة 220 passed/1 skip (test_dedup_experiment بيئي موروث: imagehash غير مثبت محليًا — CI يمرر).
