# مهمة: مراجعة معمارية لنظام أوامر موحد في مكتبة OCR

أنت **مُراجع معماري**. مهمتك فحص تصميم نظام أوامر موحد لـ `ocr-core` — مكتبة Python للتعرف الضوئي على الحروف (OCR) مع دعم عربي.

## السياق

`ocr-core` مكتبة Python (ترخيص MIT) تُغلّف محركات OCR (Tesseract, PaddleOCR, EasyOCR) وتعرض واجهة برمجية (`OCRProcessor`) لاستخراج النص من الصور مع قواعد "Visual Evidence Charter" لتحليل الرموز (✓/✗) والألوان وعدم اليقين.

**المشكلة**: المكتبة تُستخدم من 3 مشاريع مختلفة (intelli-file-manager، marathon_ted_pipeline، omni-medical-suite)، وكل مشروع يعيد اختراع واجهة استدعاء. لا يوجد عقد موحد.

**الحل المقترح**: نظام أوامر موحد مستخلص من نمط photocraft/wordcraft — "كل عملية أمر" (Command Pattern). كل أمر له id، معاملات قابلة للتسلسل، ودالة تنفيذ. يمكن تشغيل الأوامر من: Python API، CLI، HTTP، MCP.

## ما تريد مراجعته

### 1. النواة (`src/ocr_core/commands/core.py`)

- `Command` (dataclass مجمّد) — id, name, description, params_schema (JSON Schema مبسّط), run, category, requires
- `CommandRegistry` — سجل مركزي مع `register/unregister/get/list/categories`
- `@command(...)` — مزخرف للتسجيل
- `ExecutionContext` — قاموس يُمرَّر بين الأوامر لتخزين الحالة الوسيطة (image, ocr_result, corrected_text...)
- `Executor` — ينفّذ أمراً أو pipeline
- `Pipeline` — سلسلة أوامر تُبنى برمجياً أو من JSON

### 2. الأوامر المدمجة (28 أمراً في 6 فئات)

- **io** (7): load_image, load_pdf_page, save_image, save_text, list_directory, load_batch, save_snippet_crop
- **preprocess** (6): deskew, denoise, levels, curves, histogram_equalization, binarize
- **ocr** (4): extract, extract_region, extract_batch, detect_layout
- **postprocess** (5): correct_ar, visual_evidence, normalize_ar, guardrails, fuzzy_match_glossary
- **export** (4): markdown, json, docx, hf_dataset
- **benchmark** (2): cer, wer

### 3. المحوّلات

- `adapters/cli.py` — `ocr-core <list|describe|run|pipeline>`
- (مستقبلاً) `adapters/http.py` — FastAPI router
- (مستقبلاً) `adapters/mcp.py` — MCP server

### 4. الاختبارات

- `tests/commands/test_core.py` — 7 اختبارات للنواة
- `tests/commands/test_builtins.py` — 6 اختبارات للتسجيل + حساب CER/WER + حواجز الأرقام

## معايير المراجعة

### أ. صحة معمارية

1. **هل فصل "Command" و"Registry" و"Executor" و"Pipeline" منطقي؟** أم هناك تبسيط يمكن؟
2. **هل `ExecutionContext` كقاموس** هو الخيار الصحيح، أم dataclass بأسماء حقول صريحة؟
3. **هل `params_schema` بصيغة JSON Schema المبسّط** كافٍ؟ أم نحتاج pydantic؟
4. **هل التحقق اليدوي من الأنواع** (validate_params) كافٍ، أم يجب استخدام مكتبة (marshmallow/pydantic)؟
5. **نمط `@command` decorator**: هل هو الأنسب؟ أم التسجيل الصريح أفضل؟
6. **التوافق الخلفي**: هل النظام **يُضاف** إلى الكود الموجود أم **يستبدله**؟ (الهدف: صفر كسر)

### ب. جودة الكود

7. **مشاكل في الأخطاء**: ماذا لو فشل أمر في منتصف pipeline؟ هل `trace` كافٍ للتشخيص؟
8. **الأداء**: هل تمرير `ExecutionContext` عبر كل استدعاء يسبب overhead؟
9. **الاختبارات**: هل 13 اختباراً كافية لـ28 أمراً؟ ما الاختبارات الناقصة؟
10. **الترخيص**: هل هناك أي شيء في التصميم يخالف MIT أو يجلب تبعية copyleft؟

### ج. خطر التصميم (Red flags)

11. **هل هناك انتحال لأي نمط محمي بحقوق فكرية؟** (Clean Room Design)
12. **هل هناك تخزين حالة عالمي (global registry) يسبب مشاكل في الاستخدام المتوازي؟**
13. **ما الذي قد ينكسر عند تغيير `OCRProcessor.process()` توقيعها؟**

### د. اكتمال الرؤية

14. **ما الأوامر الناقصة** لاستخدام مكتبة OCR في الإنتاج؟
15. **هل MCP adapter سيكون قابلاً للتنفيذ فعلاً** بهذا التصميم؟
16. **ما الذي يجب حذفه** لأنه زائد؟

## قواعد إجابتك

- **لا تقترح مكتبات جديدة** إلا إذا كانت المراجعة تستدعي ذلك. التصميم يعتمد على stdlib فقط حاليًا.
- **لا تُعِد صياغة الكود** — راجع التصميم.
- **أشر لكل ملاحظة**: `[خطير/عالي/متوسط/منخفض]`.
- **لا تجامل**. إذا التصميم سيء في نقطة، قل ذلك بوضوح مع السبب.
- **اذكر ما هو جيد** أيضًا — ليس كل المراجعة نقدًا.

## صيغة الإخراج

### 1. الحكم التنفيذي (3-5 أسطر)
هل التصميم جاهز للتطبيق؟ نعم / لا / بعد تعديلات

### 2. قائمة الملاحظات مرتّبة بالأولوية
| # | الملاحظة | الخطورة | الموقع | الإجراء المقترح |

### 3. نقاط قوة (3-5 نقاط)

### 4. قرار صريح
- ✅ أدمج
- ⚠️ أدمج بعد تعديلات محددة (اذكرها)
- ❌ لا تدمج (مع السبب)

### 5. سؤال واحد للمصمم (لو عندك)

**ابدأ الآن. لا تسألني أسئلة توضيحية.**
