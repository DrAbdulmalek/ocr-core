# كشف الكلمات — دليل المرحلة 1

## التثبيت

```bash
# الكاشفات الأساسية (projection + heuristic)
pip install "marathon-ocr-core[preprocess]"

# الكاشف الدقيق (ONNX) — يرث [preprocess] ويضيف onnxruntime + huggingface_hub
pip install "marathon-ocr-core[word-detect]"
```

## الاستخدام السريع

### Python

```python
from PIL import Image
from ocr_core.preprocess.word_detector import detect_words
from ocr_core.preprocess.line_grouper import LineGrouper

img = Image.open("handwriting.png")
boxes = detect_words(img, kind="auto")
print(f"كشف {len(boxes)} كلمة")

grouper = LineGrouper()
lines = grouper.group(boxes, reading_direction="rtl")
print(f"في {len(lines)} سطر")
```

### عبر CLI (بعد ربط الأمر)

```bash
ocr-core run detect.words --params '{"image_key": "image"}'
```

### عبر نظام الأوامر (Python)

```python
from ocr_core.commands.core import Executor, ExecutionContext
from ocr_core.commands.builtins import io, detect

ctx = ExecutionContext()
ex = Executor()
ex.execute("io.load_image", {"path": "handwriting.png"}, ctx)
result = ex.execute("detect.words", {"detector": "projection"}, ctx)
print(result["result"]["count"], "كلمة")
```

## الكواشف

| الكاشف | السرعة | الدقة | التبعيات |
|--------|--------|-------|----------|
| `onnx` | متوسط | عالية (F1=0.88 على الإنجليزية) | onnxruntime + huggingface_hub |
| `projection` | **سريع جدًا** | جيدة للخط المتصل | لا شيء |
| `heuristic` | فوري | منخفضة | لا شيء |
| `auto` | — | يختار الأفضل المتاح | — |

## القيود المعروفة

1. **النموذج ONNX مُدرَّب على الإنجليزية (IAM)** — قد يكون أقل دقة على العربية. راجع المرحلة 2 (تدريب على بيانات عربية).
2. **`projection` يفشل على الخط المتصل جدًا** — إذا لم يكن هناك فراغات واضحة بين الكلمات.
3. **`heuristic` للإنتاج فقط في الطوارئ** — يقسّم بلا تمييز.

## الخطوات التالية (المرحلة 2)

بعد أن يعمل الكشف الأولي، سنبني **المحرر التفاعلي** (Canvas + React) لتحرير المربعات يدويًا.
