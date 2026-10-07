"""أوامر كشف الكلمات — تُسجَّل في نظام الأوامر."""
from __future__ import annotations

from ..core import command, ExecutionContext


@command(
    id="detect.words",
    name="كشف الكلمات",
    description="يكشف الكلمات في صورة مكتوبة بخط اليد أو مطبوعة.",
    category="detect",
    params_schema={
        "type": "object",
        "properties": {
            "image_key": {"type": "string"},
            "output_key": {"type": "string"},
            "detector": {"type": "string"},     # auto | onnx | projection | heuristic
            "min_confidence": {"type": "number"},
        },
        "required": [],
    },
)
def detect_words(
    ctx: ExecutionContext,
    image_key: str = "image",
    output_key: str = "word_boxes",
    detector: str = "auto",
    min_confidence: float = 0.3,
) -> dict:
    """يكشف الكلمات ويعيد قائمة BBox."""
    from ...preprocess.word_detector import create_detector

    img = ctx.get(image_key)
    if img is None:
        raise ValueError(f"لا صورة بمفتاح '{image_key}'")

    det = create_detector(detector, min_confidence=min_confidence)
    boxes = det.detect(img)

    ctx.set(output_key, boxes)
    return {
        "count": len(boxes),
        "detector": det.name,
        "avg_confidence": (
            round(sum(b.confidence for b in boxes) / len(boxes), 3)
            if boxes else 0.0
        ),
    }


@command(
    id="detect.lines",
    name="كشف السطور",
    description="يكشف الكلمات ثم يجمّعها في سطور بترتيب القراءة.",
    category="detect",
    params_schema={
        "type": "object",
        "properties": {
            "image_key": {"type": "string"},
            "boxes_key": {"type": "string"},
            "output_key": {"type": "string"},
            "reading_direction": {"type": "string"},   # rtl | ltr
            "detector": {"type": "string"},
        },
        "required": [],
    },
)
def detect_lines(
    ctx: ExecutionContext,
    image_key: str = "image",
    boxes_key: str = "word_boxes",
    output_key: str = "text_lines",
    reading_direction: str = "rtl",
    detector: str = "auto",
) -> dict:
    """يكشف الكلمات (إن لم تكن موجودة) ثم يجمّعها في سطور."""
    from ...preprocess.word_detector import create_detector
    from ...preprocess.line_grouper import LineGrouper

    # 1. احصل على المربعات
    boxes = ctx.get(boxes_key)
    if not boxes:
        img = ctx.get(image_key)
        if img is None:
            raise ValueError(f"لا صورة بمفتاح '{image_key}'")
        det = create_detector(detector)
        boxes = det.detect(img)
        ctx.set(boxes_key, boxes)

    if not boxes:
        ctx.set(output_key, [])
        return {"lines_count": 0, "words_count": 0}

    # 2. جمّع في سطور
    img = ctx.get(image_key)
    size = img.size if img else None
    grouper = LineGrouper()
    lines = grouper.group(boxes, image_size=size,
                          reading_direction=reading_direction)

    ctx.set(output_key, lines)
    return {
        "lines_count": len(lines),
        "words_count": sum(len(line) for line in lines),
    }


@command(
    id="detect.words_with_text",
    name="كشف الكلمات + OCR",
    description="يكشف الكلمات ثم يشغّل OCR على كل كلمة منفصلة.",
    category="detect",
    params_schema={
        "type": "object",
        "properties": {
            "image_key": {"type": "string"},
            "output_key": {"type": "string"},
            "detector": {"type": "string"},
            "reading_direction": {"type": "string"},
        },
        "required": [],
    },
)
def detect_words_with_text(
    ctx: ExecutionContext,
    image_key: str = "image",
    output_key: str = "word_snippets",
    detector: str = "auto",
    reading_direction: str = "rtl",
) -> dict:
    """ينتج قائمة قصاصات: كل قصاصة {bbox, text, image, line_index, word_index}."""
    from ...preprocess.word_detector import create_detector
    from ...preprocess.line_grouper import LineGrouper

    img = ctx.get(image_key)
    if img is None:
        raise ValueError(f"لا صورة بمفتاح '{image_key}'")

    # 1. اكشف الكلمات
    det = create_detector(detector)
    boxes = det.detect(img)

    if not boxes:
        ctx.set(output_key, [])
        return {"snippets": 0}

    # 2. جمّع في سطور
    grouper = LineGrouper()
    lines = grouper.group(boxes, image_size=img.size,
                          reading_direction=reading_direction)

    # 3. لكل كلمة: اقتطع صورة + شغّل OCR
    snippets = []
    for line in lines:
        for b in line:
            crop = img.crop(b.to_tuple())
            text = _ocr_word_crop(crop)
            snippets.append({
                "bbox": b.to_tuple(),
                "image": crop,
                "text": text,
                "confidence": b.confidence,
                "line_index": b.line_index,
                "word_index": b.word_index,
                "source": b.source,
            })

    ctx.set(output_key, snippets)
    return {
        "snippets": len(snippets),
        "lines": len(lines),
        "with_text": sum(1 for s in snippets if s["text"]),
    }


def _ocr_word_crop(crop) -> str:
    """OCR على قصاصة كلمة. يستخدم Tesseract إن كان مثبتًا."""
    try:
        import pytesseract
        return pytesseract.image_to_string(crop, lang="ara+eng").strip()
    except ImportError:
        return ""
    except Exception:
        return ""
