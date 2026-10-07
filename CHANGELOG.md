# سجل التغييرات — marathon-ocr-core

> ملاحظة اسم الحزمة: الاسم الحقيقي للتوزيع هو **`marathon-ocr-core`**.
> من يثبّت من git عليه استخدام هذا الاسم (وليس `ocr-core`) وإلا فشل حلّ pip:
>
> ```
> marathon-ocr-core[tesseract] @ git+https://github.com/DrAbdulmalek/ocr-core.git@v0.5.0
> ```

## غير مُصدر (فرع feat/abbyy-cleanroom-pipeline)

### أُضيف
- **`ocr_core.pipeline` — خط الأنابيب الموحد ست المراحل** (تصميم نظيف Clean-Room من
  التوثيق العام فقط — انظر `docs/research/abbyy_features.md` والقرار D-004):
  Preprocess → Layout Analysis → Recognition → Page Synthesis → Document
  Synthesis → Export، مع أوضاع تحليل التخطيط `simple/complex/tables/auto`،
  خطاف `layout_provider` للمحللات المستقبلية، سياسة النص الخام الحرفي
  (التصحيحات في مفاتيح منفصلة)، وثقة 0.0 = مجهولة بلا اختلاق.
- **`ocr_core.regional` — إعادة التعرف الانتقائي**: `extract_region` /
  `rerun_region` / `rerun_regions` بصيغتي bbox ‏(xywh / xyxy)، فشل الـ bbox
  والمحرك يُبلَّغ عبر `OCRResult.error` بلا استثناءات.
- **`ocr_core.postprocess.arabic_rtl` — `ArabicRTLPostProcessor`**: نقطة ربط
  RTL لخط الأنابيب، تفويض حرفي لـ `rtl_utils` (بلا قواعد جديدة).
- **`scripts/benchmark_ocr.sh`**: غلاف أمر واحد لمجموعة القياس الذهبية.
- **`docs/research/abbyy_features.md`**: توثيق الاستخلاص النظيف + القرار D-004
  (أفكار من التوثيق العام فقط — لا كود ولا نماذج ولا قواميس).
- اختبارات: `tests/test_pipeline.py` (20) + `tests/test_regional.py` (15) +
  `tests/test_postprocess_arabic_rtl.py` (5) = 39 اختبارًا جديدًا (المجموع 420).

## غير مُصدر (فرع feat/auto-ink-contrast)

### أُضيف
- **`ink_contrast` + توجيه التباين المنخفض في `binarize_auto`**: المقياس
  الجديد يقيس انفصال متوسطَي فئتَي Otsu (حبر/ورق) مستقلاً عن تذبذب
  الإضاءة. الموجّه يحوّل إلى Sauvola إن كان `illumination_uniformity`
  مرتفعاً **أو** `ink_contrast` دون 0.55 — يُغلق الفجوة المقاسة في العينة
  الذهبية (CER 6× على صفحات الحبر الباهت، docs/GOLDEN-SAMPLE.md §6.3)
  دون المساس بتوجيه المسح النظيف إلى Otsu والظل إلى Sauvola.

## غير مُصدر (فرع feat/golden-sample-cer)

### أُضيف
- **العينة الذهبية الموحدة** (`benchmarks/golden/`) — تنفيذ البند الأول من
  خارطة docs/09 §6: 32 زوجاً عربي/إنجليزي أصلياً حتمي التوليد (بلا صور
  مخزنة) × 4 متغيرات ثنائية (raw/otsu/sauvola/auto) × المحركات المتاحة
  بنفس التطبيع (`arabic_strong_normalize` على الطرفين) — أول جدول CER
  مقارن مملوك لنا (`docs/GOLDEN-SAMPLE.md` + `docs/golden-sample-2026-10-05/`).
  نتائج الجولة الأولى (Tesseract 5.5.0، نماذج tessdata_fast الموثقة):
  Sauvola الأفضل شمولاً (CER 0.198 مقابل 0.307 خام)؛ انهيار Otsu على
  الظل أُعيد إنتاجه قياسياً (0.553 أسوأ من الخام 0.535) وSauvola/auto
  يخفضانه 54%؛ الموجّه auto وجّه الظل صحيحاً وفوّت التباين المنخفض
  (فجوة موثقة بمقاييس).
- رندر عربي عبر **libraqm/HarfBuzz** (نص منطقي خام، RTL طبيعي) — يصل
  CER إلى 0.0000 على الصفحات النظيفة، مقابل ~0.80 معكوس الترتيب مع مسار
  reshaper+bidi الكلاسيكي (موثق في `docs/GOLDEN-SAMPLE.md` §3).
- خط Noto Naskh Arabic مرفق داخل الحزمة (رخصة SIL OFL 1.1 —
  `benchmarks/golden/fonts/`) لضمان حتمية الرندر عبر البيئات.
- إضافة `golden` إلى `optional-dependencies` (arabic-reshaper +
  python-bidi) وتثبيتها في CI.

## غير مُصدر (فرع feat/algos-docs09)

### أُضيف
- **حزمة ثنائية مرجعية** (`preprocess/binarize.py`) من التعريفات المنشورة حصرًا:
  - `otsu_threshold` / `binarize_otsu` — العتبة العالمية المُعظِّمة للتباين
    بين الفئتين (Otsu 1979)، متجهة عبر cumsum لكل 256 عتبة، مع سقوط آمن
    للصور الفارغة.
  - `sauvola_thresholds` / `binarize_sauvola` — العتبة المحلية التكيفية
    (Sauvola & Pietikäinen 2000) عبر **صور التكامل** بكلفة O(h·w) مستقلة
    عن حجم النافذة — تتعامل مع الظلال والإضاءة غير المتجانسة التي تُسقط Otsu.
  - `illumination_uniformity` + `binarize_auto` — موجّه تلقائي يقيس تذبذب
    الإضاءة منخفض التردد (معامل تباين متوسطات كتل الصفحة) ويحوّل الصفحة:
    مسح نظيف → Otsu، كاميرا بظلال → Sauvola (قاعدة docs/09 §2.2).
  - كل الرياضيات numpy صرف؛ cv2 اختياري للتحويل الرمادي فقط.
- **حزمة فك تسلسل CTC** (`decoding/ctc.py`) من التعريف المنشور حصرًا
  (Graves et al., ICML 2006):
  - `ctc_greedy_decode` — فك جشع O(T) (argmax → دمج التكرارات → حذف الفراغ).
  - `ctc_greedy_decode_confident` — **ثقة لكل حرف** (متوسط posterior على
    إطارات الحرف + نطاق الإطارات) لتلبية شرط §5.2-3 (علامات تحقق كهرمانية
    في محرر المراجعة).
  - `ctc_beam_search_decode` — بحث بادئات كلاسيكي (p_b/p_nb لكل بادئة،
    إعادة تطبيع كل إطار، تقليم أعلى N فئة لكل إطار) — يتفوق على الجشع
    عند التوزيعات الملتبسة.
  - يقبل احتمالات مُطبّعة أو logits (softmax تلقائي مستقر عدديًا).
- ربط في `enhance_for_ocr`/`fix_scan`: معامل `binarize_method`
  ("adaptive" الافتراضي يحفظ سلوك الاستدعاءات القائمة | "otsu" | "sauvola" | "auto")
  عبر `apply_binarization`.
- اختبارات: `tests/test_binarize.py` (12 حالة) و`tests/test_ctc.py` (14 حالة).

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
