# نظام الأوامر — ocr-core

مستخلص من نمط photocraft/wordcraft: **كل عملية أمر**.

## الاستخدام السريع

```bash
pip install -e ".[dev]"

# اسرد الأوامر
ocr-core list

# وصف أمر
ocr-core describe ocr.extract

# نفّذ أمراً
ocr-core run io.load_image --params '{"path": "/tmp/scan.png"}'
```

## مثال خط أنابيب

`pipeline.json`:
```json
{
  "name": "arabic_ocr",
  "steps": [
    {"command": "io.load_image", "params": {"path": "/tmp/scan.png"}},
    {"command": "preprocess.deskew"},
    {"command": "ocr.extract"},
    {"command": "postprocess.correct_ar"},
    {"command": "export.markdown", "params": {"output_path": "/tmp/out.md"}}
  ]
}
```

```bash
ocr-core pipeline pipeline.json
```

## البرمجة

```python
from ocr_core.commands.core import Executor, Pipeline, ExecutionContext
from ocr_core.commands.builtins import io, preprocess, ocr, postprocess, export

p = (Pipeline("demo")
     .add("io.load_image", path="scan.png")
     .add("ocr.extract")
     .add("postprocess.correct_ar"))

result = Executor().execute_pipeline(p)
print(result["ok"], result["trace"])
```

## الفئات

- **io**: تحميل/حفظ الصور والملفات
- **preprocess**: deskew, denoise, levels, curves, histogram, binarize
- **ocr**: extract, extract_region, extract_batch, detect_layout
- **postprocess**: correct_ar, visual_evidence, normalize_ar, guardrails, fuzzy_match
- **export**: markdown, json, docx, hf_dataset
- **benchmark**: cer, wer
- **detect** (المرحلة 1): words, lines, words_with_text — كشف كلمات الخط اليدوي
- **vlm_ocr** (المرحلة 4 — تُسجَّل في فئة ocr): vlm.extract, vlm.extract_batch, vlm.pdf_to_markdown — jina-ocr-v1

> ملاحظة تدقيق: `ocr.extract` يغلّف المحركات الفعلية (`engines/*` —
> `process_image`) ثم يمرر النص إلى `OCRProcessor.process_text` (ميثاق 18
> قاعدة). سياسة الثقة محفوظة: `confidence == 0.0` = غير معروفة ولا تُمرر
> للميثاق كثقة منخفضة.

