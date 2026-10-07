# jina-ocr-v1 — محرك VLM للمستندات

## ⚠️ تنبيه ترخيصي

- **الكود في ocr-core**: MIT (آمن تجاريًا)
- **الأوزان على HuggingFace**: **CC BY-NC 4.0** (غير تجاري)

لاستخدام تجاري: تحتاج ترخيصًا من Jina/Elastic.

## التثبيت

```bash
pip install "marathon-ocr-core[jina]"
```

يحمّل: `torch` + `transformers` + `torchvision` + `accelerate` + النموذج (~7GB).

## المواصفات

| الميزة | القيمة |
|--------|--------|
| المعمارية | MoE (DeepSeek-OCR backbone) |
| المعاملات الكلية | 3.4B |
| النشطة | 574M/token |
| VRAM | ~7.4 GB (FP16) |
| OmniDocBench v1.6 | 91.14 |
| olmOCR-Bench | 83.4 |
| Throughput | 2.57 ص/ث (A100 c=32) |
| اللغات | 100+ |

## المُخرج

- **Markdown** للنص العادي
- **HTML** للجداول
- **LaTeX** للمعادلات
- **لا bounding boxes** — استخدم محركًا آخر للكلمات الفردية

## الاستخدام

### Python مباشر

```python
from ocr_core.engines.jina_vlm import JinaVLMEngine, STRICT_OCR_PROMPT

engine = JinaVLMEngine()
result = engine.extract("document.png", prompt=STRICT_OCR_PROMPT)
print(result.markdown)
```

### عبر نظام الأوامر

```python
from ocr_core.commands.core import Executor, ExecutionContext
from ocr_core.commands.builtins import vlm_ocr  # noqa

ctx = ExecutionContext()
Executor().execute("vlm.extract", {
    "image_path": "doc.png",
    "strict": True,
}, ctx)
print(ctx.get("vlm_markdown"))
```

### PDF كامل → Markdown

```python
Executor().execute("vlm.pdf_to_markdown", {
    "pdf_path": "book.pdf",
    "output_path": "book.md",
    "dpi": 200,
}, ctx)
```

### CLI

```bash
ocr-core run vlm.extract --params '{"image_path": "doc.png"}'
ocr-core pipeline pipelines/pdf_to_md.json
```

## قيود الاستخدام

### ❌ لا يصلح لـ

- **قصاصات الخط اليدوي**: لا يعطي bbox
- **الاستخدام التجاري**: CC BY-NC 4.0
- **CPU فقط**: بطيء جدًا (< 1 صفحة/دقيقة)
- **تدريب TrOCR**: يحتاج بيانات مجزأة

### ✅ مثالي لـ

- **PDF أكاديمي** → Markdown منسّق
- **تقارير طبية** بجداول
- **مستندات بمعدلات**
- **استخراج نص كامل** بسرعة (2.57 ص/ث على GPU)

## الاختبارات

```bash
# اختبارات الوحدة (سريعة، بلا GPU)
pytest tests/engines/test_jina_vlm.py -v

# اختبارات الأوامر
pytest tests/commands/test_vlm_commands.py -v

# اختبارات التكامل
pytest tests/integration/test_vlm_integration.py -v

# اختبار حقيقي (GPU + نموذج)
RUN_VLM_TESTS=1 pytest tests/engines/test_jina_vlm.py::TestRealInference -v

# سكربت حيوي
python scripts/test_vlm_quick.py --arabic --strict
```

## الموارد

- [Model card](https://huggingface.co/jinaai/jina-ocr-v1)
- [Paper](https://arxiv.org/abs/2609.03181)
- [Jina Blog](https://jina.ai/news/jina-ocr-v1-faster-document-parsing-on-low-budget-gpus/)
- [Elastic Blog](https://www.elastic.co/search-labs/blog/ocr-model-jina-ocr-v1-document-parsing)
