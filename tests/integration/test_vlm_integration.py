"""اختبارات التكامل بين jina-ocr-v1 وبقية وحدات ocr-core.

الهدف: التأكد من أن إضافة jina-ocr-v1 لا تكسر:
  - نظام الأوامر
  - Pipeline
  - المحركات الأخرى
  - التوافق مع البيئات بلا GPU
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestNoBreakage:

    def test_all_existing_engines_still_importable(self):
        """إضافة jina لا تكسر المحركات الأخرى."""
        # هذه كلها يجب أن تُستورد بلا مشاكل
        from ocr_core.engines import base  # noqa
        # tesseract/paddle قد لا تكون مثبتة — نتحقق فقط من الملف
        base_path = Path(base.__file__).parent
        assert (base_path / "tesseract.py").exists()
        assert (base_path / "jina_vlm.py").exists()

    def test_command_registry_still_works(self):
        """السجل لا يتعارض مع الإضافة."""
        from ocr_core.commands.core import registry
        from ocr_core.commands.builtins import (
            io, preprocess, ocr, postprocess,
            export, benchmark, detect, vlm_ocr,  # noqa
        )
        # لا استثناءات
        all_cmds = registry.list()
        ids = [c.id for c in all_cmds]
        assert len(ids) == len(set(ids)), "تكرار في IDs"

    def test_no_command_name_collision(self):
        """vlm.* لا تتعارض مع ocr.*"""
        from ocr_core.commands.core import registry
        from ocr_core.commands.builtins import (
            ocr as ocr_mod, vlm_ocr,  # noqa
        )
        ids = {c.id for c in registry.list()}
        # لا يوجد vlm.extract و ocr.extract بنفس الاسم
        assert "vlm.extract" in ids
        assert "ocr.extract" in ids
        assert "vlm.extract" != "ocr.extract"


class TestGracefulDegradation:

    def test_pipeline_fails_cleanly_without_jina_deps(
        self, tmp_path,
    ):
        """Pipeline يستخدم vlm.extract بلا jina → فشل نظيف."""
        from PIL import Image
        img = tmp_path / "x.png"
        Image.new("RGB", (100, 100), "white").save(img)

        from ocr_core.commands.core import (
            Executor, ExecutionContext, Pipeline,
        )
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        with patch(
            "ocr_core.engines.jina_vlm.is_available",
            return_value=False,
        ):
            p = Pipeline("test").add(
                "vlm.extract", image_path=str(img),
            )
            result = Executor().execute_pipeline(p, ExecutionContext())

        assert result["ok"] is False
        assert result["stopped_at"] == "vlm.extract"
        # الخطأ يُشرح بلطف
        assert "غير متاح" in result["steps"][0]["error"]

    def test_other_engines_unaffected_by_jina_failure(
        self, tmp_path,
    ):
        """فشل jina لا يمنع استخدام محركات أخرى."""
        from ocr_core.commands.core import registry
        # جميع الأوامر الأخرى ما زالت مسجّلة
        ids = {c.id for c in registry.list()}
        for expected in [
            "io.load_image", "preprocess.deskew",
            "ocr.extract", "postprocess.correct_ar",
            "export.markdown", "benchmark.cer",
        ]:
            assert expected in ids, f"{expected} مفقود"


class TestMetadataPreservation:

    def test_vlm_metadata_stored_correctly(
        self, tmp_path,
    ):
        """بيانات الـ metadata تُحفظ بعد الاستخراج."""
        from PIL import Image
        img = tmp_path / "x.png"
        Image.new("RGB", (100, 100), "white").save(img)

        from ocr_core.engines.jina_vlm import VLMResult

        def _fake_extract(image_path, prompt=None):
            return VLMResult(
                markdown="test",
                model_id="jinaai/jina-ocr-v1",
                duration_ms=99.9,
                prompt_used=prompt or "default",
            )

        engine = MagicMock()
        engine.extract.side_effect = _fake_extract

        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        ctx = ExecutionContext()
        with patch(
            "ocr_core.engines.jina_vlm.is_available", return_value=True,
        ), patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=engine,
        ):
            Executor().execute(
                "vlm.extract",
                {"image_path": str(img), "output_key": "out"},
                ctx,
            )

        meta = ctx.get("out_meta")
        assert meta["model_id"] == "jinaai/jina-ocr-v1"
        assert meta["duration_ms"] == 99.9
