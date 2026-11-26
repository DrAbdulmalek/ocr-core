# سجل التغييرات — marathon-ocr-core

> ملاحظة اسم الحزمة: الاسم الحقيقي للتوزيع هو **`marathon-ocr-core`**.
> من يثبّت من git عليه استخدام هذا الاسم (وليس `ocr-core`) وإلا فشل حلّ pip:
>
> ```
> marathon-ocr-core[tesseract] @ git+https://github.com/DrAbdulmalek/ocr-core.git@v0.5.0
> ```

## غير مُصدر (فرع fix/offline-local-only-model-resolution)

### أُضيف
- **سياسة الوصول للنماذج `ocr_core.model_policy` (issue #79، تتبّع #77 و IFM#89)**:
  مفردات بلا أي تبعيات — `LOCAL_ONLY` (الافتراضي)، `EXPLICIT_REMOTE`
  (الشبكة فقط مع `remote_authorized=True`)، `REMOTE_DISABLED`؛ المحاكم:
  بارامتر المُنشئ > بيئة (`OCR_CORE_MODEL_POLICY`/`OCR_CORE_REMOTE_AUTHORIZED`)
  > الافتراضي، ووضع غير معروف يفشل مغلقًا (`ValueError`) بلا تخمين صامت،
  و`ModelArtifactMissingError` برسالة قابلة للتنفيذ عند نقص الخزنة.

### مُعدَّل
- **حل النماذج محلي-فقط افتراضيًا**: `OnnxWordDetector` و`JinaVLMEngine`
  يحلّان الأثر من الخزنة المحلية أولًا (`local_files_only=True`)، ونقص
  الأثر يفشل مغلقًا بلا أي محاولة شبكة ولا إعادة محاولة بعيدة؛ الشبكة
  محاولة واحدة صريحة فقط مع `EXPLICIT_REMOTE` مصرّح (منع cache-miss
  retries بالتصميم).
- **المسار الآلي لكواشف الكلمات**: `create_detector("auto")` يتراجع إلى
  `ProjectionDetector` عند غياب الأثر المحلي مع منع البُعد (ومفاتيح
  onnx الخاصة تُفلتر عن projection)؛ الاختيار الصريح `kind="onnx"` يبقى
  فشلًا مغلقًا عند نقص الأثر.
- **15 اختبارًا جديدًا** (`tests/test_model_policy.py`): استيراد النواة
  بلا huggingface_hub/transformers/torch/onnxruntime/requests (مفسّر
  معزول `-I` بجذر استيراد src فقط)، جدول التصريح الصريح، سلوك نقص
  الخزنة بلا إعادة محاولة (stubs تسجّل النداءات)، الـfallback المحلي
  للمسار الآلي، وفشل jina المغلق — كلها اصطناعية بلا شبكة وبلا أوزان.

## غير مُصدر (فرع feat/contract-gate-73)

### أُضيف
- **بوابة العقد (issue #73)**: بصمات ذهبية للعقد — `tests/test_contract.py` + `tests/golden_contract_signature.json` (ملف ذاتي التحقق بـsha256) تقفل: حقول `OCRResult` dataclass (الاسم/النمط/الافتراضي)، `schema_version` والحقول المطلوبة للمخطط (أعلى المستوى + الكلمة)، المثال النموذجي (= `VALID_RESULT` حرفيًا — لا مصدرين للحقيقة)، دلالات الثقة 0-1 وbbox xywh، وترخيص MIT. أي انحراف = فشل الاختبار ويوجب تحديث البصمة في نفس الـPR.
- **حارس CI**: `.github/workflows/contract-guard.yml` + `scripts/ci/contract_guard.py` (stdlib فقط) — أي PR يمس سطح العقد (المخطط، base.py، CONTRACT.md، ملفات البصمة) يفشل ما لم يرفع الإصدار الرئيسي في `pyproject.toml`؛ والكسر إضافيًا يوجب رفع `schema_version` الرئيسي وإشعارًا مسبقًا للمستهلكين الثلاثة.
- `docs/CONTRACT.md`: قسم «سياسة التغيير» (القواعد الأربع للبوابة)، قسم «المثال النموذجي» (مستند schema-1.0 كامل)، وجدول المستهلكين (toolkit/AMH/MRA + التزام common-core المستقبلي بالاستهلاك لا النسخ).

### مُرقّى
- **إصدار الحزمة `0.7.0` → `1.0.0`**: البوابة تُطبَّق على نفسها — أول PR يمس سطح العقد رفع الإصدار الرئيسي وفق قاعدتها الجديدة، ومها إشارة أن العقد صار رسميًا ومستقرًا (v1.0.0 = عقد مثبت ببصمات ذهبية وحارس CI).

## غير مُصدر (فرع roadmap/ocr-result-schema)

### أُضيف
- **عقد `OCRResult` الموحّد صار مخطط JSON صريحًا ومُسوّى (draft-07، `schema_version` ثابت "1.0")** — `src/ocr_core/schemas/ocr_result.schema.json` مع محمّل `ocr_core.schemas.load_schema()` (importlib.resources + مسار احتياطي)، وثيقة العقد `docs/CONTRACT.md` (الحقول المطلوبة، سياسة الثقة 0-1 وصفر = مجهولة، هندسة bbox بصيغة xywh، جسر GPL/AGPL عبر تبادل ملفات فقط)، واختبارات `tests/test_ocr_result_schema.py` (15 اختبارًا) — الأولوية 8 من خارطة الطريق المدمجة: المحركات ومصوّت الـensemble ومشاهد الاختلاف manyhands والفهرس يقرأون بنية واحدة بلا محوّلات.

## غير مُصدر (فرع fix/import-shadow-arabic-render)

### أُصلح
- **حزمة الظل القديمة `ocr_core/` (v0.4.0) في جذر المستودع حُذفت نهائياً** —
  كانت تعترض الاستيراد وتعيد أي مستهلك يستورد من الجذر إلى مسار الـpre-bidi
  القديم (الذي وثّق العيّنة الذهبية أنه ينتج ترتيباً معكوساً ≈0.79 CER)،
  وَجَعلت اختبارات مثل `test_rtl_utils.py` تنهار عند تشغيلها منفردة
  (كانت تعمل فقط بحظّ ترتيب التحصيل حين يسبقها ملف يحقن `src` في `sys.path`).
  مرجع الميثاق `charter_v2.yaml` نُقل إلى `docs/` (الميثاق الحقيقي المربوط
  بـsha يبقى في `src/ocr_core/config/` — لا تغيير).
- **إصلاح عزم الاستيراد**: `[tool.pytest.ini_options] pythonpath = ["src"]`
  في `pyproject.toml` — كل الاختبارات تُحلّ الحزمة الحقيقية v0.7.0 حتماً،
  بلا اعتماد على حيل `sys.path.insert` الفردية (بقيت لكنها أصبحت احتياطية).
- **Pillow 13**: إسقاط `mode="L"` من `Image.fromarray` في
  `benchmarks/golden/render.py` و`runner.py` (الوسيط أُلغي في Pillow 13 —
  الاستدلال من dtype يعطي "L" نفسها).

### التحقق
- `pytest`: **504 passed, 7 skipped** (سوت كامل أخضر بعد الإصلاح).
- اختبارات كانت منهارة منفردة (`test_rtl_utils.py`,
  `test_postprocess_arabic_rtl.py`) تعمل الآن وحدها.
- فحص بصري لمسار raqm: رسم «مرحبا بالعالم الطبي» بحروف متصلة وترتيب RTL
  صحيح من الخط المرفق Noto Naskh.

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


## غير مُصدر — verified ModelStore engine integration

- ربط OnnxWordDetector وJinaVLMEngine اختياريًا بخزنة ModelStore عبر model_store_root وmodel_revision غير القابل للتغيير.
- ONNX يحل الملف المعلن فقط عبر get_verified_file؛ Jina يحل لقطة النموذج الكاملة عبر get_verified. فشل التحقق لا يؤدي إلى HF أو تنزيل بديل.
- أضيفت اختبارات اصطناعية لمسارات الربط ومتطلبات revision؛ لا أوزان ولا بيانات مرضى.


### مُحسّن — ملكية lease أثناء التشغيل

- يحتفظ محركا ONNX وJina بقفل `ModelStore.lease()` طوال عمر الجلسة/النموذج المحمّل.
- أضيفت `close()` لتحرير القفل وتفريغ مراجع المحرك؛ وعند فشل التحميل يُحرر القفل قبل إعادة رفع الخطأ.
- اختبارات اصطناعية تتحقق من بقاء القفل أثناء التشغيل وتحريره عند `close()`.


### أمان إضافي — Jina custom code

- جعل `trust_remote_code=False` افتراضيًا في `JinaVLMEngine`، حتى مع لقطة ModelStore موثقة.
- يتطلب تنفيذ كود النموذج المخصص تمرير `trust_remote_code=True` صراحةً بعد مراجعة الكود نفسه؛ سلامة SHA-256 لا تعني سلامة الكود.
- اختبارات اصطناعية تتحقق من الرفض الافتراضي ومن مسار التصريح الصريح.


### أداة تحقق محلية للأوزان الحقيقية

- إضافة `scripts/smoke_test_model_store.py` لتشغيل تحميل واستدلال محلي على صورة بيضاء اصطناعية فقط، مع تقرير عدّادات/أزمنة دون طباعة النص المستخرج أو الوصول للشبكة.
- يدعم ONNX وJina؛ تنفيذ كود Jina المخصص يتطلب `--trust-remote-code` صراحةً. الأداة اختبار تحميل/تشغيل فقط وليست قياس دقة أو تحققًا سريريًا.


### توثيق مراجعة بطاقات النماذج

- توثيق أن PellelNitram WordDetector كاشف صناديق كلمات فقط (ليس recognizer)، وبطاقة النموذج تذكر IAM وتحدّ اختلاف أنماط الكتابة.
- توثيق أن Jina OCR v1 مرخّص CC BY-NC 4.0 (غير تجاري)، 3B parameters، ويطلب upstream تشغيل custom code؛ التكامل هنا يبقيه معطلًا افتراضيًا.
- لا توجد دعوى دقة للخط الطبي العربي أو المستندات السريرية السورية.

### تقوية حدود مسارات ModelStore

- رفض الروابط الرمزية في مكونات مسارات registry وartifact، لا في المسار النهائي فقط.
- التحقق من بنية registry دون تعديل عند القراءة، والتحقق قبل إنشاء المجلدات الفرعية لتجنب الكتابة عبر root رمزي.
- إضافة اختبارات انحدار للمجلدات الرمزية؛ لم تتغير واجهة OCRResult أو صيغة manifest.

### تحسين أداة التحقق المحلية

- فرض أعلام Hugging Face وTransformers للعمل دون اتصال قبل استيراد مكتبات النماذج، وتعطيل telemetry.
- فصل زمن تحميل النموذج عن زمن الاستدلال الأول، وإظهار ذروة RSS للعملية.
- إضافة اختبارات لضبط بيئة offline وقياس RSS؛ الحماية على مستوى التطبيق وليست بديلاً عن عزل الشبكة على مستوى نظام التشغيل.
