"""محرك jina-ocr-v1 — VLM end-to-end للمستندات الكاملة.

⚠️ ترخيص الأوزان: CC BY-NC 4.0 (غير تجاري).
    الكود في ocr-core MIT؛ الأوزان تُحمّل عند التشغيل من HuggingFace.

الاختلاف عن Tesseract/Paddle:
    - لا يعيد bounding boxes
    - يعيد Markdown كامل (نص + جداول + LaTeX)
    - يحتاج GPU (~7GB VRAM بـ FP16)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "jinaai/jina-ocr-v1"
DEFAULT_PROMPT = (
    "Transcribe the provided document image into a clean "
    "Markdown format, preserving the natural reading order."
)

# Prompt أكاديمي أكثر دقة (يجبر LaTeX + HTML tables)
STRICT_OCR_PROMPT = (
    "Just return the plain text representation of this document "
    "as if you were reading it naturally. Turn equations and math "
    "symbols into a LaTeX representation. Convert tables into HTML "
    "format. Remove the headers and footers, but keep references "
    "and footnotes. Read any natural handwriting."
)


@dataclass
class VLMResult:
    """نتيجة استخراج VLM."""
    markdown: str
    model_id: str
    duration_ms: float = 0.0
    prompt_used: str = ""

    def to_dict(self) -> dict:
        return {
            "markdown": self.markdown,
            "model_id": self.model_id,
            "duration_ms": round(self.duration_ms, 2),
            "prompt_used": self.prompt_used,
        }


class JinaVLMEngine:
    """محرك jina-ocr-v1 — مستقل، lazy-loading."""

    name = "jina-vlm"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 4096,
        cache_dir: Optional[Path] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir

        self._model = None
        self._processor = None
        self._device = None

    def _ensure_loaded(self):
        """تحميل كسول — لا يُحمّل حتى أول استخراج."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError(
                "jina-ocr-v1 يتطلب تبعيات ثقيلة. ثبّت:\n"
                "    pip install 'marathon-ocr-core[jina]'"
            )

        # تحديد الجهاز
        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        logger.info("تحميل %s على %s...", self.model_id, self._device)

        # ⚠️ trust_remote_code=True إلزامي لهذا النموذج
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        ).to(self._device)

        self._model.eval()
        logger.info("✅ %s جاهز", self.model_id)

    def extract(
        self,
        image_path: str,
        prompt: Optional[str] = None,
    ) -> VLMResult:
        """استخراج Markdown من صورة مستند."""
        import time

        # تحقق مبكر من وجود الملف — قبل تحميل النموذج (7GB).
        # (إصلاح تدقيق: كان المستخدم ينتظر التحميل الكامل ثم يفشل
        #  بـ FileNotFoundError من PIL — راجع ملاحظة الاختبارات.)
        if not Path(image_path).exists():
            raise FileNotFoundError(f"الصورة غير موجودة: {image_path}")

        self._ensure_loaded()

        import torch
        from PIL import Image

        start = time.perf_counter()

        img = Image.open(image_path).convert("RGB")
        used_prompt = prompt or self.prompt

        # تجهيز المدخلات
        inputs = self._processor.prepare_ocr_inputs(
            img,
            prompt=used_prompt,
            device=self._device,
        )

        # التوليد
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # فك التشفير
        markdown = self._processor.decode_ocr(
            output, inputs["input_ids"]
        )
        duration = (time.perf_counter() - start) * 1000

        return VLMResult(
            markdown=markdown.strip(),
            model_id=self.model_id,
            duration_ms=duration,
            prompt_used=used_prompt,
        )

    def extract_with_confidence(self, image_path: str) -> dict:
        """jina-ocr-v1 لا يعيد confidences — نعيد عنصرًا وهميًا للتوافق."""
        result = self.extract(image_path)
        return {
            "text": result.markdown,
            "confidence": 0.9,        # ثقة ثابتة (لا تتوفر درجة حقيقية)
            "engine": self.name,
            "note": "VLM engine — no per-word confidence available",
        }


def is_available() -> bool:
    """هل يمكن استخدام jina-ocr-v1 في هذه البيئة؟"""
    try:
        import torch  # noqa
        import transformers  # noqa
        return True
    except ImportError:
        return False
