"""أوامر OCR عبر VLM — لـ jina-ocr-v1 وغيره."""
from __future__ import annotations

from pathlib import Path
from ..core import command, ExecutionContext


@command(
    id="vlm.extract",
    name="استخراج Markdown من صورة",
    description="يستخدم VLM (jina-ocr-v1) لتحويل صورة مستند إلى Markdown.",
    category="ocr",
    params_schema={
        "type": "object",
        "properties": {
            "image_path": {"type": "string"},
            "output_key": {"type": "string"},
            "prompt": {"type": "string"},
            "strict": {"type": "boolean"},
        },
        "required": ["image_path"],
    },
)
def vlm_extract(
    ctx: ExecutionContext,
    image_path: str = "",
    output_key: str = "vlm_markdown",
    prompt: str = "",
    strict: bool = False,
) -> dict:
    """استخراج Markdown عبر jina-ocr-v1.

    Args:
        image_path: مسار صورة (PNG مفضّل، 1024×1024).
        strict: استخدم prompt أكاديمي (LaTeX + HTML tables).
    """
    from ...engines.jina_vlm import (
        JinaVLMEngine, STRICT_OCR_PROMPT, is_available,
    )

    if not is_available():
        raise RuntimeError(
            "jina-ocr-v1 غير متاح. ثبّت التبعيات:\n"
            "    pip install 'marathon-ocr-core[jina]'"
        )

    if not Path(image_path).exists():
        raise FileNotFoundError(f"الصورة غير موجودة: {image_path}")

    used_prompt = prompt
    if strict:
        used_prompt = STRICT_OCR_PROMPT

    engine = JinaVLMEngine()
    result = engine.extract(image_path, prompt=used_prompt or None)

    ctx.set(output_key, result.markdown)
    ctx.set(f"{output_key}_meta", result.to_dict())

    return {
        "markdown_length": len(result.markdown),
        "model": result.model_id,
        "duration_ms": round(result.duration_ms, 1),
    }


@command(
    id="vlm.extract_batch",
    name="استخراج دفعة عبر VLM",
    description="يعالج عدة صور عبر jina-ocr-v1 ويجمع النتائج.",
    category="ocr",
    params_schema={
        "type": "object",
        "properties": {
            "images": {"type": "array"},
            "output_dir": {"type": "string"},
            "strict": {"type": "boolean"},
        },
        "required": ["images", "output_dir"],
    },
)
def vlm_extract_batch(
    ctx: ExecutionContext,
    images: list = None,
    output_dir: str = "",
    strict: bool = False,
) -> dict:
    """يعالج قائمة صور ويحفظ Markdown لكل منها."""
    from ...engines.jina_vlm import (
        JinaVLMEngine, STRICT_OCR_PROMPT,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    engine = JinaVLMEngine()
    prompt = STRICT_OCR_PROMPT if strict else None

    results = []
    for img_path in images or []:
        try:
            r = engine.extract(img_path, prompt=prompt)
            md_file = out / f"{Path(img_path).stem}.md"
            md_file.write_text(r.markdown, encoding="utf-8")
            results.append({
                "image": img_path,
                "markdown_file": str(md_file),
                "length": len(r.markdown),
                "ok": True,
            })
        except Exception as e:
            results.append({
                "image": img_path,
                "ok": False,
                "error": str(e),
            })

    ctx.set("vlm_batch_results", results)
    return {
        "total": len(results),
        "ok": sum(1 for r in results if r["ok"]),
    }


@command(
    id="vlm.pdf_to_markdown",
    name="تحويل PDF إلى Markdown",
    description="يحوّل PDF كامل (كل الصفحات) إلى Markdown عبر jina-ocr-v1.",
    category="ocr",
    params_schema={
        "type": "object",
        "properties": {
            "pdf_path": {"type": "string"},
            "output_path": {"type": "string"},
            "dpi": {"type": "integer"},
            "strict": {"type": "boolean"},
        },
        "required": ["pdf_path", "output_path"],
    },
)
def vlm_pdf_to_markdown(
    ctx: ExecutionContext,
    pdf_path: str = "",
    output_path: str = "",
    dpi: int = 200,
    strict: bool = False,
) -> dict:
    """PDF → صور → jina-ocr-v1 → Markdown موحّد."""
    import io

    # إصلاح تدقيق: تحقق من وجود الملف *قبل* استيراد fitz — حتى لا تظهر
    # رسالة "No module named 'fitz'" بدل "PDF غير موجود" عند خطأ المسار.
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF غير موجود: {pdf_path}")

    try:
        import fitz
    except ImportError:
        raise ImportError(
            "vlm.pdf_to_markdown يتطلب pymupdf. ثبّت:\n"
            "    pip install 'marathon-ocr-core[pdf]' (أو [vlm])"
        )

    from PIL import Image
    from ...engines.jina_vlm import (
        JinaVLMEngine, STRICT_OCR_PROMPT,
    )

    engine = JinaVLMEngine()
    prompt = STRICT_OCR_PROMPT if strict else None

    doc = fitz.open(pdf_path)
    parts: list[str] = []
    pages_processed = 0

    try:
        for page_num, page in enumerate(doc):
            # حوّل الصفحة إلى صورة
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # احفظ مؤقتًا
            tmp = Path(f"/tmp/_jina_page_{page_num}.png")
            img.save(tmp)

            try:
                r = engine.extract(str(tmp), prompt=prompt)
                parts.append(f"<!-- Page {page_num + 1} -->\n\n{r.markdown}")
                pages_processed += 1
            finally:
                tmp.unlink(missing_ok=True)
    finally:
        doc.close()

    full_markdown = "\n\n---\n\n".join(parts)
    Path(output_path).write_text(full_markdown, encoding="utf-8")
    ctx.set("vlm_full_markdown", full_markdown)

    return {
        "pages": pages_processed,
        "output": output_path,
        "total_length": len(full_markdown),
    }
