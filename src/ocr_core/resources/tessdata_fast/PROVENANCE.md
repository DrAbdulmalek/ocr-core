# Tessdata Resources — Provenance (المنشأ والرخصة)

## ما هذا؟

نماذج Tesseract المدرَّبة الجاهزة للتعرف على **العربية والإنجليزية**، مضمّنة هنا لتعمل محليًا بلا تنزيل خارجي. هذه هي **البديل المفتوح والمرخص** لأي نماذج مملوكة (انظر docs في مستودع تحليل FineReader APK: نماذج ABBYY .rom مملوكة ومحمية بترخيص RSA — لا تُنزَّل ولا يُعاد توزيعها).

## المصدر

| البند | القيمة |
|---|---|
| المستودع | https://github.com/tesseract-ocr/tessdata_fast |
| التزام المصدر | `87416418657359cb625c412a48b6e1d6d41c29bd` (2024-08-01) |
| الملفان | `ara.traineddata` (1,432,056 بايت)، `eng.traineddata` (4,113,088 بايت) |
| الرخصة | **Apache-2.0** (https://github.com/tesseract-ocr/tessdata_fast/blob/main/LICENSE) |

## البصمات (SHA-256) — للتحقق بعد أي نسخ/تنزيل

```
e3206d3dc87fd50c24a0fb9f01838615911d25168f4e64415244b67d2bb3e729  ara.traineddata
7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2  eng.traineddata
```

## لماذا tessdata_fast وليس tessdata_best؟

- `fast`: نسخة int8 مخففة (~1.4MB عربي) — أسرع 3-4× بفارق دقة مقبول للنصوص الواضحة؛ مناسبة كافتراضي مضمن.
- `best` (LSTM float عالي الدقة): أكبر (~2.5MB+ عربي) — للجودة القصوى. `scripts/fetch_tessdata.py --best` يجلبها عند الحاجة إلى مجلد محلي (خارج الحزمة).

## الاستخدام

```python
from ocr_core.engines.tesseract import TesseractEngine

# الافتراضي: يكتشف تلقائيًا النماذج المضمنة ويستخدم --tessdata-dir
engine = TesseractEngine(lang="ara+eng")
result = engine.process_image("page.png")
print(result.text)

# توجيه مجلد نماذج مخصص (مثل best المنزّل):
engine = TesseractEngine(lang="ara+eng", tessdata_dir="/path/to/tessdata_best")
```

لا يحتاج صلاحيات شبكة عند التشغيل؛ التنزيل يحدث فقط عبر `scripts/fetch_tessdata.py` أثناء التطوير.

## ملاحظة أمانة

`ara.traineddata` هنا هو جودة "fast" — لتطبيقاتنا الطبية/الماراثونية عالية الدقة، المسار المعتمد هو **التدريب المخصص** من قصاصات محرر edit-ocr (JSONL → LSTM fine-tune)، وهذا الملف مجرد أساس عمل فوري.
