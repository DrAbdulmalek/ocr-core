"""أوامر OCR — تغلّف محركات ocr_core.engines + OCRProcessor (ميثاق الـ 18 قاعدة).

ملاحظة تدقيق (2026-10): النسخة الأصلية من المحادثة كانت تستدعي
``OCRProcessor.process(path)`` — وهي واجهة غير موجودة. العقد الفعلي:
  - المحركات (Tesseract/EasyOCR/Paddle): ``process_image(path) -> OCRResult``
  - الميثاق: ``OCRProcessor.process_text(raw_text, confidence) -> dict``
هذا الملف يصلح الربط بينهما مع الحفاظ على سياسة الثقة:
``confidence == 0.0`` تعني "غير معروفة" — لا تُختلق أبدًا، ولا تُمرَّر
إلى الميثاق كثقة منخفضة (وإلا وسم R15 كل الكلمات بـ [؟؟]).
"""
from __future__ import annotations

import os
import tempfile

from ..core import command, ExecutionContext
from ...rules.engine import OCRProcessor


def _default_engine():
    """يختار أول محرك متاح: Tesseract ← EasyOCR ← Paddle.

    الاستيرادات محمية: كل محرك extra اختياري مستقل.
    Returns None عندما لا يتوفر أي محرك.
    """
    candidates = []
    try:
        from ...engines.tesseract import TesseractEngine
        candidates.append(TesseractEngine())
    except Exception:
        pass
    try:
        from ...engines.easyocr import EasyOCREngine
        candidates.append(EasyOCREngine())
    except Exception:
        pass
    try:
        from ...engines.paddle import PaddleEngine
        candidates.append(PaddleEngine())
    except Exception:
        pass
    for eng in candidates:
        try:
            if eng.available():
                return eng
        except Exception:
            continue
    return None


def _run_engine_on_image(img, engine=None, lang: str | None = None) -> dict:
    """يحفظ الصورة مؤقتًا ويشغّل المحرك — يعيد OCRResult كـ dict."""
    engine = engine or _default_engine()
    if engine is None:
        raise RuntimeError(
            "لا محرك OCR متاح — ثبّت أحد الإضافات: "
            "pip install 'marathon-ocr-core[tesseract]' (أو [easyocr] / [paddle])"
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        img.save(tmp.name)
        result = engine.process_image(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if not result.ok:
        raise RuntimeError(f"المحرك '{engine.name}' فشل: {result.error}")
    return result


@command(
    id="ocr.extract", name="استخراج نص",
    description="يشغّل OCR على الصورة في السياق ثم قواعد الميثاق (18 قاعدة).",
    category="ocr",
    params_schema={"type": "object", "properties": {
        "image_key": {"type": "string"}, "output_key": {"type": "string"},
        "rules_file": {"type": "string"},
        "apply_charter": {"type": "boolean"}}, "required": []},
)
def extract(ctx: ExecutionContext, image_key: str = "image",
            output_key: str = "ocr_result", rules_file: str = None,
            apply_charter: bool = True) -> dict:
    img = ctx.get(image_key)
    if img is None:
        raise ValueError(f"لا صورة بمفتاح '{image_key}'")

    ocr_res = _run_engine_on_image(img)

    result = {
        "text": ocr_res.text,
        "raw_text": ocr_res.text,
        "engine": ocr_res.engine,
        # 0.0 = غير معروفة (سياسة ocr-core) — لا تُختلق درجة ثقة أبدًا
        "confidence": ocr_res.confidence,
        "markers": [],
        "uncertain_flags": 0,
    }

    if apply_charter:
        proc = OCRProcessor(rules_file) if rules_file else OCRProcessor()
        # سياسة الثقة: 0.0 تعني "غير معروفة" — تمرَّر 1.0 للميثاق حتى لا
        # يسم R15 كل الكلمات كغير مؤكدة بسبب مجهولية الثقة.
        conf = ocr_res.confidence if ocr_res.confidence > 0.0 else 1.0
        processed = proc.process_text(ocr_res.text, conf)
        result["text"] = processed["markdown"]
        result["correct"] = processed["correct"]
        result["wrong"] = processed["wrong"]
        result["ratio"] = processed["ratio"]
        result["metadata"] = processed["metadata"]
        result["markers"] = processed["metadata"].get("markers", [])
        result["uncertain_flags"] = processed["metadata"].get("uncertain_flags", 0)

    ctx.set(output_key, result)
    return {
        "text_length": len(result["text"]),
        "engine": result["engine"],
        "markers_count": len(result["markers"]),
        "uncertain_flags": result["uncertain_flags"],
    }


@command(
    id="ocr.extract_region", name="استخراج منطقة",
    description="OCR على منطقة محددة (bbox).",
    category="ocr",
    params_schema={"type": "object", "properties": {
        "image_key": {"type": "string"}, "bbox": {"type": "array"},
        "output_key": {"type": "string"}}, "required": ["bbox"]},
)
def extract_region(ctx: ExecutionContext, image_key: str = "image",
                   bbox: list = None, output_key: str = "ocr_result") -> dict:
    if not bbox or len(bbox) != 4:
        raise ValueError("bbox يجب أن يكون [x1, y1, x2, y2]")
    img = ctx.get(image_key)
    if img is None:
        raise ValueError(f"لا صورة بمفتاح '{image_key}'")
    crop = img.crop(tuple(bbox))
    ctx.set("_region_image", crop)
    return extract(ctx, image_key="_region_image", output_key=output_key)


@command(
    id="ocr.extract_batch", name="استخراج دفعة",
    description="OCR على عدة صور من قائمة.",
    category="ocr",
    params_schema={"type": "object", "properties": {
        "batch_key": {"type": "string"}, "output_key": {"type": "string"}},
        "required": []},
)
def extract_batch(ctx: ExecutionContext, batch_key: str = "images",
                  output_key: str = "ocr_results") -> dict:
    batch = ctx.get(batch_key, [])
    results = []
    for item in batch:
        if "image" not in item:
            continue
        ctx.set("_batch_img", item["image"])
        try:
            extract(ctx, image_key="_batch_img", output_key="_batch_out")
            results.append({"path": item.get("path"), "ok": True,
                            "result": ctx.get("_batch_out")})
        except Exception as e:
            results.append({"path": item.get("path"), "ok": False, "error": str(e)})
    ctx.set(output_key, results)
    return {"total": len(batch), "ok": sum(1 for r in results if r["ok"])}


@command(
    id="ocr.detect_layout", name="كشف التخطيط",
    description="يكشف مناطق النص/الصور/الجداول (تحليل بسيط).",
    category="ocr",
    params_schema={"type": "object", "properties": {
        "image_key": {"type": "string"}, "output_key": {"type": "string"}},
        "required": []},
)
def detect_layout(ctx: ExecutionContext, image_key: str = "image",
                  output_key: str = "layout") -> dict:
    img = ctx.get(image_key)
    if img is None:
        raise ValueError(f"لا صورة بمفتاح '{image_key}'")
    w, h = img.size
    regions = []
    try:
        import numpy as np
        arr = np.array(img.convert("L"))
        row_has_content = (arr < 128).sum(axis=1) > (arr.shape[1] * 0.01)
        in_block = False
        start = 0
        for i, has in enumerate(row_has_content):
            if has and not in_block:
                start = i
                in_block = True
            elif not has and in_block:
                if i - start > 10:
                    regions.append({"y1": int(start), "y2": int(i), "type": "text"})
                in_block = False
        if in_block:
            regions.append({"y1": int(start), "y2": int(len(row_has_content)),
                            "type": "text"})
    except ImportError:
        regions = [{"y1": 0, "y2": h, "type": "unknown"}]
    result = {"width": w, "height": h, "regions": regions}
    ctx.set(output_key, result)
    return {"regions_count": len(regions)}
