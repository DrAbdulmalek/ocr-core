"""اختبارات أوامر VLM — vlm.extract، vlm.extract_batch، vlm.pdf_to_markdown."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# فيكسترات
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (300, 150), "white")
    path = tmp_path / "test.png"
    img.save(path)
    return str(path)


@pytest.fixture
def sample_pdf(tmp_path):
    """PDF بسيط من صفحتين."""
    try:
        import fitz
    except ImportError:
        pytest.skip("pymupdf غير مثبت")

    doc = fitz.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} content", fontsize=12)
    path = tmp_path / "sample.pdf"
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture
def fake_engine():
    """محرك وهمي — يعيد Markdown ثابتًا بدون تحميل النموذج."""
    from ocr_core.engines.jina_vlm import VLMResult

    def _fake_extract(image_path: str, prompt=None) -> VLMResult:
        return VLMResult(
            markdown=f"# Fake\n\nContent of {Path(image_path).name}",
            model_id="mock-jina",
            duration_ms=10.0,
            prompt_used=prompt or "default",
        )

    engine = MagicMock()
    engine.extract.side_effect = _fake_extract
    engine.name = "jina-vlm-mock"
    return engine


# ---------------------------------------------------------------------------
# 1. التسجيل
# ---------------------------------------------------------------------------

class TestVLMCommandRegistration:

    def test_vlm_commands_registered(self):
        from ocr_core.commands.builtins import vlm_ocr  # noqa
        from ocr_core.commands.core import registry
        ids = {c.id for c in registry.list()}
        assert "vlm.extract" in ids
        assert "vlm.extract_batch" in ids
        assert "vlm.pdf_to_markdown" in ids

    def test_vlm_commands_in_ocr_category(self):
        from ocr_core.commands.builtins import vlm_ocr  # noqa
        from ocr_core.commands.core import registry
        cmds = registry.list(category="ocr")
        ids = {c.id for c in cmds}
        assert "vlm.extract" in ids

    def test_vlm_extract_schema_has_image_path(self):
        from ocr_core.commands.builtins import vlm_ocr  # noqa
        from ocr_core.commands.core import registry
        cmd = registry.get("vlm.extract")
        props = cmd.params_schema.get("properties", {})
        assert "image_path" in props
        assert props["image_path"]["type"] == "string"


# ---------------------------------------------------------------------------
# 2. vlm.extract
# ---------------------------------------------------------------------------

class TestVLMExtract:

    def test_fails_clearly_when_deps_missing(self, sample_image):
        """بلا jina مثبت → خطأ واضح."""
        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        with patch(
            "ocr_core.engines.jina_vlm.is_available",
            return_value=False,
        ):
            result = Executor().execute(
                "vlm.extract",
                {"image_path": sample_image},
                ExecutionContext(),
            )
            assert result["ok"] is False
            assert "غير متاح" in result["error"]
            assert "jina" in result["error"].lower()

    def test_fails_on_missing_file(self):
        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        with patch(
            "ocr_core.engines.jina_vlm.is_available",
            return_value=True,
        ):
            result = Executor().execute(
                "vlm.extract",
                {"image_path": "/nonexistent/file.png"},
                ExecutionContext(),
            )
            assert result["ok"] is False
            assert "غير موجود" in result["error"]

    def test_successful_extract_stores_markdown(
        self, sample_image, fake_engine,
    ):
        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        ctx = ExecutionContext()

        with patch(
            "ocr_core.engines.jina_vlm.is_available",
            return_value=True,
        ), patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=fake_engine,
        ):
            result = Executor().execute(
                "vlm.extract",
                {"image_path": sample_image},
                ctx,
            )

        assert result["ok"] is True
        assert result["result"]["markdown_length"] > 0
        assert ctx.get("vlm_markdown").startswith("# Fake")
        assert ctx.get("vlm_markdown_meta")["model_id"] == "mock-jina"

    def test_custom_prompt_passed_to_engine(
        self, sample_image, fake_engine,
    ):
        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        ctx = ExecutionContext()
        with patch(
            "ocr_core.engines.jina_vlm.is_available", return_value=True,
        ), patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=fake_engine,
        ):
            Executor().execute(
                "vlm.extract",
                {"image_path": sample_image, "prompt": "custom prompt"},
                ctx,
            )

        # تحقق أن المحرك استقبل الـ prompt
        call_args = fake_engine.extract.call_args
        assert call_args[1]["prompt"] == "custom prompt"

    def test_strict_mode_uses_strict_prompt(
        self, sample_image, fake_engine,
    ):
        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa
        from ocr_core.engines.jina_vlm import STRICT_OCR_PROMPT

        ctx = ExecutionContext()
        with patch(
            "ocr_core.engines.jina_vlm.is_available", return_value=True,
        ), patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=fake_engine,
        ):
            Executor().execute(
                "vlm.extract",
                {"image_path": sample_image, "strict": True},
                ctx,
            )

        call_args = fake_engine.extract.call_args
        assert call_args[1]["prompt"] == STRICT_OCR_PROMPT

    def test_output_key_customizable(self, sample_image, fake_engine):
        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        ctx = ExecutionContext()
        with patch(
            "ocr_core.engines.jina_vlm.is_available", return_value=True,
        ), patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=fake_engine,
        ):
            Executor().execute(
                "vlm.extract",
                {"image_path": sample_image, "output_key": "my_key"},
                ctx,
            )

        assert ctx.get("my_key") is not None
        assert "my_key_meta" in ctx.data


# ---------------------------------------------------------------------------
# 3. vlm.extract_batch
# ---------------------------------------------------------------------------

class TestVLMExtractBatch:

    def test_batch_processes_multiple_images(
        self, tmp_path, fake_engine,
    ):
        from PIL import Image
        images = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            Image.new("RGB", (100, 50), "white").save(p)
            images.append(str(p))

        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        ctx = ExecutionContext()
        out_dir = tmp_path / "out"
        with patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=fake_engine,
        ):
            result = Executor().execute(
                "vlm.extract_batch",
                {"images": images, "output_dir": str(out_dir)},
                ctx,
            )

        assert result["ok"] is True
        assert result["result"]["total"] == 3
        assert result["result"]["ok"] == 3

    def test_batch_creates_md_files(self, tmp_path, fake_engine):
        from PIL import Image
        img = tmp_path / "test.png"
        Image.new("RGB", (100, 50), "white").save(img)

        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        out_dir = tmp_path / "out"
        with patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=fake_engine,
        ):
            Executor().execute(
                "vlm.extract_batch",
                {"images": [str(img)], "output_dir": str(out_dir)},
                ExecutionContext(),
            )

        md_files = list(out_dir.glob("*.md"))
        assert len(md_files) == 1
        assert md_files[0].read_text(encoding="utf-8").startswith("# Fake")

    def test_batch_handles_failures_gracefully(self, tmp_path, fake_engine):
        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        # اجعل المحرك يفشل على الأول فقط
        def _fail_first(image_path, prompt=None):
            if "fail" in image_path:
                raise RuntimeError("محاكاة فشل")
            from ocr_core.engines.jina_vlm import VLMResult
            return VLMResult(markdown="ok", model_id="mock")

        fake_engine.extract.side_effect = _fail_first

        ctx = ExecutionContext()
        with patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=fake_engine,
        ):
            result = Executor().execute(
                "vlm.extract_batch",
                {
                    "images": ["/tmp/fail.png", "/tmp/ok.png"],
                    "output_dir": str(tmp_path / "out"),
                },
                ctx,
            )

        assert result["result"]["total"] == 2
        assert result["result"]["ok"] == 1
        batch = ctx.get("vlm_batch_results")
        assert batch[0]["ok"] is False
        assert batch[1]["ok"] is True


# ---------------------------------------------------------------------------
# 4. vlm.pdf_to_markdown
# ---------------------------------------------------------------------------

class TestVLMPdfToMarkdown:

    def test_pdf_not_found(self):
        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        result = Executor().execute(
            "vlm.pdf_to_markdown",
            {"pdf_path": "/nope.pdf", "output_path": "/tmp/out.md"},
            ExecutionContext(),
        )
        assert result["ok"] is False
        assert "غير موجود" in result["error"]

    def test_pdf_success_multipage(self, sample_pdf, tmp_path, fake_engine):
        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        out_md = tmp_path / "result.md"
        ctx = ExecutionContext()

        with patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=fake_engine,
        ):
            result = Executor().execute(
                "vlm.pdf_to_markdown",
                {
                    "pdf_path": sample_pdf,
                    "output_path": str(out_md),
                    "dpi": 100,
                },
                ctx,
            )

        assert result["ok"] is True
        assert result["result"]["pages"] == 2
        assert out_md.exists()
        content = out_md.read_text(encoding="utf-8")
        assert "<!-- Page 1 -->" in content
        assert "<!-- Page 2 -->" in content

    def test_pdf_writes_full_markdown_to_context(
        self, sample_pdf, tmp_path, fake_engine,
    ):
        from ocr_core.commands.core import Executor, ExecutionContext
        from ocr_core.commands.builtins import vlm_ocr  # noqa

        ctx = ExecutionContext()
        with patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=fake_engine,
        ):
            Executor().execute(
                "vlm.pdf_to_markdown",
                {
                    "pdf_path": sample_pdf,
                    "output_path": str(tmp_path / "out.md"),
                },
                ctx,
            )

        assert ctx.get("vlm_full_markdown") is not None
        assert len(ctx.get("vlm_full_markdown")) > 0


# ---------------------------------------------------------------------------
# 5. التكامل مع Pipeline
# ---------------------------------------------------------------------------

class TestVLMPipeline:

    def test_pipeline_combines_vlm_with_export(
        self, sample_image, tmp_path, fake_engine,
    ):
        from ocr_core.commands.core import (
            Executor, ExecutionContext, Pipeline,
        )
        from ocr_core.commands.builtins import vlm_ocr  # noqa
        from ocr_core.commands.builtins import export  # noqa
        from ocr_core.commands.builtins import io  # noqa

        out_md = tmp_path / "pipeline_out.md"

        with patch(
            "ocr_core.engines.jina_vlm.is_available", return_value=True,
        ), patch(
            "ocr_core.engines.jina_vlm.JinaVLMEngine",
            return_value=fake_engine,
        ):
            pipeline = (
                Pipeline("vlm_export")
                .add("vlm.extract",
                     image_path=sample_image,
                     output_key="markdown_text")
                .add("export.markdown",
                     text_key="markdown_text",
                     output_path=str(out_md),
                     title="Test Doc")
            )
            result = Executor().execute_pipeline(pipeline, ExecutionContext())

        assert result["ok"] is True
        assert out_md.exists()


# ---------------------------------------------------------------------------
# 6. التوافق الخلفي
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_existing_ocr_command_still_works(self, sample_image):
        """vlm.extract لا يكسر ocr.extract."""
        from ocr_core.commands.core import registry
        from ocr_core.commands.builtins import ocr  # noqa
        assert "ocr.extract" in registry
        assert "vlm.extract" in registry
        # كلاهما في نفس الفئة
        ids = {c.id for c in registry.list(category="ocr")}
        assert "ocr.extract" in ids
        assert "vlm.extract" in ids

    def test_command_count_increased(self):
        """عدد الأوامر يجب أن يزيد بـ3 على الأقل."""
        from ocr_core.commands.core import registry
        from ocr_core.commands.builtins import vlm_ocr  # noqa
        # العدد الإجمالي يجب أن يكون ≥ 31
        assert len(registry.list()) >= 31
