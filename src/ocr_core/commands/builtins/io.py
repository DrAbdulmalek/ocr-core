from pathlib import Path
from ..core import command, ExecutionContext


@command(
    id="io.load_image", name="تحميل صورة",
    description="يحمّل صورة من القرص إلى السياق (PIL.Image).",
    category="io",
    params_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)
def load_image(ctx: ExecutionContext, path: str) -> dict:
    from PIL import Image
    img = Image.open(path)
    ctx.set("image", img)
    ctx.set("image_path", path)
    return {"path": path, "size": img.size, "mode": img.mode}


@command(
    id="io.load_pdf_page", name="تحميل صفحة PDF",
    description="يحوّل صفحة PDF إلى صورة في السياق.",
    category="io",
    params_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "page": {"type": "integer"},
            "dpi": {"type": "integer"},
        },
        "required": ["path"],
    },
)
def load_pdf_page(ctx: ExecutionContext, path: str,
                   page: int = 0, dpi: int = 300) -> dict:
    import fitz
    from PIL import Image
    import io as _io
    doc = fitz.open(path)
    if page >= len(doc):
        raise ValueError(f"الصفحة {page} غير موجودة (إجمالي {len(doc)})")
    pix = doc[page].get_pixmap(dpi=dpi)
    img = Image.open(_io.BytesIO(pix.tobytes("png")))
    ctx.set("image", img)
    ctx.set("pdf_path", path)
    ctx.set("pdf_page", page)
    return {"page": page, "pages_total": len(doc),
            "size": img.size, "dpi": dpi}


@command(
    id="io.save_image", name="حفظ صورة",
    description="يحفظ الصورة من السياق إلى القرص.",
    category="io",
    params_schema={
        "type": "object",
        "properties": {
            "context_key": {"type": "string"},
            "output_path": {"type": "string"},
            "format": {"type": "string"},
        },
        "required": ["output_path"],
    },
)
def save_image(ctx: ExecutionContext, context_key: str = "image",
                output_path: str = "", format: str = "PNG") -> dict:
    img = ctx.get(context_key)
    if img is None:
        raise ValueError(f"لا صورة بمفتاح '{context_key}'")
    img.save(output_path, format=format)
    return {"path": output_path, "bytes": Path(output_path).stat().st_size}


@command(
    id="io.save_text", name="حفظ نص",
    description="يحفظ نصاً من السياق إلى ملف.",
    category="io",
    params_schema={
        "type": "object",
        "properties": {
            "context_key": {"type": "string"},
            "output_path": {"type": "string"},
            "encoding": {"type": "string"},
        },
        "required": ["context_key", "output_path"],
    },
)
def save_text(ctx: ExecutionContext, context_key: str,
               output_path: str, encoding: str = "utf-8") -> dict:
    val = ctx.get(context_key, "")
    text = val if isinstance(val, str) else (val.get("text", "") if isinstance(val, dict) else str(val))
    Path(output_path).write_text(text, encoding=encoding)
    return {"path": output_path, "bytes": len(text.encode(encoding))}


@command(
    id="io.list_directory", name="اسرد ملفات",
    description="يعرض قائمة ملفات مع فلترة امتداد.",
    category="io",
    params_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "extensions": {"type": "array"},
        },
        "required": ["path"],
    },
)
def list_directory(ctx: ExecutionContext, path: str,
                    extensions: list = None) -> dict:
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"ليس مجلداً: {path}")
    files = []
    for f in p.iterdir():
        if f.is_file():
            if extensions and f.suffix.lower() not in [e.lower() for e in extensions]:
                continue
            files.append(str(f))
    ctx.set("file_list", files)
    return {"count": len(files), "files": files[:50]}


@command(
    id="io.load_batch", name="تحميل دفعة صور",
    description="يحمّل عدة صور دفعة واحدة إلى قائمة.",
    category="io",
    params_schema={
        "type": "object",
        "properties": {
            "paths": {"type": "array"},
            "output_key": {"type": "string"},
        },
        "required": ["paths"],
    },
)
def load_batch(ctx: ExecutionContext, paths: list,
                output_key: str = "images") -> dict:
    from PIL import Image
    imgs = []
    for p in paths:
        try:
            imgs.append({"path": p, "image": Image.open(p)})
        except Exception as e:
            imgs.append({"path": p, "error": str(e)})
    ctx.set(output_key, imgs)
    return {"loaded": sum(1 for i in imgs if "image" in i),
            "failed": sum(1 for i in imgs if "error" in i)}


@command(
    id="io.save_snippet_crop", name="حفظ قصاصة",
    description="يقتطع منطقة من الصورة ويحفظها كقصاصة تدريب.",
    category="io",
    params_schema={
        "type": "object",
        "properties": {
            "image_key": {"type": "string"},
            "bbox": {"type": "array"},
            "output_path": {"type": "string"},
        },
        "required": ["bbox", "output_path"],
    },
)
def save_snippet_crop(ctx: ExecutionContext, image_key: str = "image",
                       bbox: list = None, output_path: str = "") -> dict:
    img = ctx.get(image_key)
    if img is None:
        raise ValueError("لا صورة في السياق")
    if not bbox or len(bbox) != 4:
        raise ValueError("bbox يجب أن يكون [x1, y1, x2, y2]")
    crop = img.crop(tuple(bbox))
    crop.save(output_path)
    return {"path": output_path, "size": crop.size, "bbox": bbox}
