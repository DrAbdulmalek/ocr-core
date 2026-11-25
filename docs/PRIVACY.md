# سياسة الخصوصية — طبقة البيانات الطبية (OCR-04 / ocr-core#76)

> تنفيذ قواعد الأمان للطبقة الخاصة (خطة الدمج §8.3). **بيانات طبية: هذه
> مسؤولية قانونية/أخلاقية لا ترف.** هذه الوثيقة حاكمة لوحدة
> [`ocr_core.privacy`](../src/ocr_core/privacy.py) ولكل ما يمس بيانات المرضى.

## 1. القاعدة المقصورة على الوحدة (غير قابلة للتفاوض)

**بيانات المرضى وصور التقارير أبدًا في git** — HF private/gated datasets
أو محليًا فقط. لا API خارجي، لا OCR سحابي، لا شبكة في أي مسار يلمس محتوى
طبيًا. المخالفات المطلقة:

| المكان | الحكم |
|---|---|
| git (أي مستودع، أي فرع) | ❌ ممنوع مطلقًا |
| HF dataset عام | ❌ ممنوع مطلقًا |
| HF dataset private أو gated | ✅ مسموح (بعد المراجعة اليدوية) |
| محليًا على جهاز المالك | ✅ مسموح |
| خدمة سحابية (OCR/تصحيح/ترجمة) | ❌ ممنوع مطلقًا |

## 2. حالة مجموعة التصحيحات — إجراء عاجل معلق على المالك

`DrAbdulmalek/arabic-medical-ocr-corrections` (319 سجلًا مستخرجة من تقارير
طبية يدوية) **كان عامًا وغير gated وقت كتابة هذا الملف** (27 تنزيلًا موثقة
في القضية #76). الإجراء المطلوب فورًا — أمر واحد:

```bash
# آليًا (يتطلب HF_TOKEN بصلاحية كتابة — يُقرأ من البيئة فقط، لا يُطبع):
HF_TOKEN=hf_xxx python3 scripts/hf_gate_dataset.py                # gated=manual
HF_TOKEN=hf_xxx python3 scripts/hf_gate_dataset.py --mode private # إخفاء تام
python3 scripts/hf_gate_dataset.py --verify-only                  # تحقق بعد التنفيذ
```

يدويًا (بدون سكربت): صفحة الـdataset → Settings → تفعيل **Gating**
(Request access = manual) أو **Private**. المهم: لا يبقى عامًا.

## 3. اختبار السجلات الحقيقية (معيار القبول)

معيار القبول «`ocr_core.privacy` يجتاز اختبار الـ10 سجلات» يُنفَّذ على جهاز
المالك محليًا — السجلات الحقيقية لا تدخل git ولا CI ولا أي بيئة وكيل:

```bash
python3 scripts/privacy_selftest.py --records /path/to/corrections.jsonl
python3 scripts/privacy_selftest.py --records x.jsonl --out report.json --ner
```

السكربت يطبع **عدّادات فقط** (حجب/بقايا/حكم) ولا يطبع أي محتوى خام من
السجلات؛ التقرير المولد عُدّادات حصرًا ويُرفق في تعليق على #76.

## 4. القرارات المعلقة على المالك (لا تُغلق القضية بدونها)

1. **تنفيذ gating** (البند 2 أعلاه) + إلغاء أي توكن HF مستخدم في الجلسات.
2. **المراجعة اليدوية الموثقة للسجلات الـ319** قبل أي إعادة نشر — المراجع:
   المالك.
3. **قرار تكلفة HF**: هل private dataset صغير يكفي لمخرجات `crops/` أم يلزم
   Bucket؟ ومن يدفع؟ — يوثق في تعليق على #76.

## 5. حدود الوحدة (توثيق الاستعادة)

المصدر: وحدة `hybrid_scrubber.py` من مستودع `medical-ocr-training-hub`
المؤرشف. الفروق موثقة في ترويسة [`src/ocr_core/privacy.py`](../src/ocr_core/privacy.py):
حذف `logging.basicConfig` (نظافة مكتبة)، إضافة `scrub_records` و
`check_residue` وثوابت السياسة، وطبقة NER تحميل مؤجَّل معطلة افتراضيًا في
CI/الاختبارات (لا شبكة). الاختبارات كلها بسجلات اصطناعية خيالية.


## NER offline enforcement

The optional Arabic NER pipeline is local-only. Model resolution passes
`local_files_only=True`; a cache miss disables NER gracefully and keeps the
regex layer available. It must not fetch model weights from Hugging Face or
any other network source. This setting does not replace owner review of any
model files already present on disk.


### تفسير نتائج self-test (تنبيه مهم)

لا يجوز اعتبار غياب مطابقات Regex دليلًا على خلو السجلات من الأسماء أو
كل أنواع المعلومات الشخصية. لذلك:
- دون `--ner`: النتيجة `regex_only_incomplete_manual_review_required` ورمز خروج 2.
- مع `--ner` لكن دون نموذج محلي محمّل: النتيجة غير مكتملة ورمز خروج 2؛ لا تنزيل شبكي.
- مع NER محلي نشط ودون بقايا Regex: النتيجة تعني «لم تُكتشف بقايا ضمن هذه الطبقات» فقط، ورمز خروج 0، مع بقاء المراجعة البشرية إلزامية.
- أي بقايا Regex تعطي رمز خروج 1.

حتى NER النشط ليس ضمانًا شاملًا لإزالة PII؛ لا إعادة نشر قبل مراجعة المالك
يدويًا للسجلات الـ319 وتوثيق القرار.
