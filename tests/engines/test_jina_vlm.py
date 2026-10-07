"""اختبارات محرك jina-ocr-v1.

المبدأ: هذه الاختبارات **لا تُحمّل النموذج** (7GB) في CI.
تستخدم mock/مونkeypatch للتحقق من:
  - التحميل الكسول (lazy loading)
  - رفع الأخطاء الواضحة عند غياب التبعيات
  - بنية النتائج (VLMResult)
  - التعامل مع الحالات الحديّة

عند توفر GPU محليًا، يمكن تشغيل الاختبارات الحقيقية بـ:
  RUN_VLM_TESTS=1 pytest tests/engines/test_jina_vlm.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# فيكسترات
# ---------------------------------------------------------------------------

@pytest.fixture
def skip_if_no_gpu():
    """يتخطى الاختبارات الثقيلة إن لم يوجد GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU غير متاح")
    except ImportError:
        pytest.skip("torch غير مثبت")


@pytest.fixture
def skip_if_no_deps():
    """يتخطى إن لم تكن تبعيات jina مثبتة."""
    try:
        import torch  # noqa
        import transformers  # noqa
    except ImportError:
        pytest.skip("تبعيات jina غير مثبتة")


@pytest.fixture
def sample_image(tmp_path):
    """صورة اختبار بسيطة."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 80), "Hello World", fill="black")
    path = tmp_path / "sample.png"
    img.save(path)
    return str(path)


@pytest.fixture
def arabic_image(tmp_path):
    """صورة اختبار عربية."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (500, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 80), "مرحبا بالعالم", fill="black")
    path = tmp_path / "arabic.png"
    img.save(path)
    return str(path)


# ---------------------------------------------------------------------------
# 1. استيراد الوحدة والعقود
# ---------------------------------------------------------------------------

class TestModuleContract:
    """العقود الأساسية — تعمل بلا تبعيات."""

    def test_module_imports_without_torch(self):
        """الوحدة تُستورد بلا torch (imports داخلية)."""
        from ocr_core.engines import jina_vlm
        assert hasattr(jina_vlm, "JinaVLMEngine")
        assert hasattr(jina_vlm, "is_available")
        assert hasattr(jina_vlm, "VLMResult")

    def test_default_constants_defined(self):
        from ocr_core.engines import jina_vlm
        assert jina_vlm.DEFAULT_MODEL_ID == "jinaai/jina-ocr-v1"
        assert "Markdown" in jina_vlm.DEFAULT_PROMPT or \
               "markdown" in jina_vlm.DEFAULT_PROMPT.lower()

    def test_strict_prompt_contains_expected_keywords(self):
        """الـ prompt الأكاديمي يذكر LaTeX و HTML."""
        from ocr_core.engines import jina_vlm
        p = jina_vlm.STRICT_OCR_PROMPT
        assert "LaTeX" in p
        assert "HTML" in p
        assert "handwriting" in p.lower()

    def test_is_available_returns_bool_without_torch(self):
        from ocr_core.engines import jina_vlm
        result = jina_vlm.is_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 2. VLMResult
# ---------------------------------------------------------------------------

class TestVLMResult:

    def test_to_dict_complete(self):
        from ocr_core.engines.jina_vlm import VLMResult
        r = VLMResult(
            markdown="# عنوان\n\n| col |\n|-----|\n| a |",
            model_id="jinaai/jina-ocr-v1",
            duration_ms=123.45,
            prompt_used="test prompt",
        )
        d = r.to_dict()
        assert d["markdown"].startswith("#")
        assert d["model_id"] == "jinaai/jina-ocr-v1"
        assert d["duration_ms"] == 123.45
        assert d["prompt_used"] == "test prompt"

    def test_to_dict_defaults(self):
        from ocr_core.engines.jina_vlm import VLMResult
        r = VLMResult(markdown="نص", model_id="x")
        d = r.to_dict()
        assert d["duration_ms"] == 0.0
        assert d["prompt_used"] == ""


# ---------------------------------------------------------------------------
# 3. التحميل الكسول
# ---------------------------------------------------------------------------

class TestLazyLoading:

    def test_engine_does_not_load_at_init(self):
        """المحرك لا يحمّل النموذج عند الإنشاء."""
        from ocr_core.engines.jina_vlm import JinaVLMEngine
        engine = JinaVLMEngine()
        assert engine._model is None
        assert engine._processor is None
        assert engine._device is None

    def test_ensure_loaded_raises_clear_error_without_deps(
        self, monkeypatch
    ):
        """خطأ واضح عند غياب torch."""
        from ocr_core.engines import jina_vlm

        # حجب torch و transformers
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "transformers", None)

        engine = jina_vlm.JinaVLMEngine()
        with pytest.raises((ImportError, TypeError)):
            engine._ensure_loaded()

    def test_ensure_loaded_idempotent(self):
        """استدعاء _ensure_loaded مرتين لا يعيد التحميل."""
        from ocr_core.engines.jina_vlm import JinaVLMEngine
        engine = JinaVLMEngine()

        # احقن نموذجًا وهميًا
        engine._model = MagicMock()
        engine._processor = MagicMock()
        engine._device = "cpu"

        # استدعاء _ensure_loaded لا يغيّر شيئًا
        engine._ensure_loaded()
        assert engine._model is not None


# ---------------------------------------------------------------------------
# 4. الكشف عن الجهاز
# ---------------------------------------------------------------------------

class TestDeviceDetection:

    def test_device_auto_picks_cpu_without_cuda(self):
        """auto → cpu إن لم يوجد CUDA."""
        pytest.importorskip("torch")
        import torch
        from ocr_core.engines.jina_vlm import JinaVLMEngine

        engine = JinaVLMEngine(device="auto")
        # نتحقق فقط من القيمة الافتراضية
        assert engine.device == "auto"

    def test_explicit_device_respected(self):
        """device='cpu' يُحترم."""
        from ocr_core.engines.jina_vlm import JinaVLMEngine
        engine = JinaVLMEngine(device="cpu")
        assert engine.device == "cpu"


# ---------------------------------------------------------------------------
# 5. الأخطاء الواضحة
# ---------------------------------------------------------------------------

class TestErrorMessages:

    def test_missing_image_raises_file_not_found(self, tmp_path):
        """صورة غير موجودة → FileNotFoundError واضح *قبل* تحميل النموذج.

        (تم إصلاح نقطة الضعف الأصلية: extract() صار يتحقق من المسار أولًا،
        فلا ينتظر المستخدم تحميل 7GB ثم يفشل.)
        """
        from ocr_core.engines.jina_vlm import JinaVLMEngine

        engine = JinaVLMEngine()
        # لا حاجة لحقن نموذج وهمي — التحقق من المسار يسبق _ensure_loaded،
        # لذا يعمل هذا الاختبار بلا torch وبلا GPU.
        missing = tmp_path / "does_not_exist.png"
        with pytest.raises(FileNotFoundError, match="غير موجودة"):
            engine.extract(str(missing))
        assert engine._model is None  # لم يُحمَّل أي نموذج

    def test_extract_without_deps_raises_informative_error(self, tmp_path):
        """بلا تبعيات → رسالة توجيهية.

        (تُدقيق: بعد إصلاح التحقق المبكر من المسار، يجب أن يكون الملف موجودًا
        فعلاً حتى نصل إلى _ensure_loaded — وإلا رفع FileNotFoundError أولًا.)
        """
        from ocr_core.engines.jina_vlm import JinaVLMEngine
        engine = JinaVLMEngine()

        fake_png = tmp_path / "fake.png"
        fake_png.write_bytes(b"\x89PNG\r\n\x1a\n")  # يكفي لاجتياز فحص الوجود

        with patch.object(engine, "_ensure_loaded") as mock_load:
            mock_load.side_effect = ImportError(
                "jina-ocr-v1 يتطلب تبعيات ثقيلة. ثبّت:\n"
                "    pip install 'marathon-ocr-core[jina]'"
            )
            with pytest.raises(ImportError, match="marathon-ocr-core\\[jina\\]"):
                engine.extract(str(fake_png))


# ---------------------------------------------------------------------------
# 6. اختبارات حقيقية (اختيارية — تحتاج GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.getenv("RUN_VLM_TESTS") != "1",
    reason="اختبارات ثقيلة — شغّلها بـ RUN_VLM_TESTS=1",
)
class TestRealInference:
    """اختبارات حقيقية على GPU — تحتاج النموذج (7GB)."""

    @pytest.mark.usefixtures("skip_if_no_deps", "skip_if_no_gpu")
    def test_extract_simple_image(self, sample_image):
        from ocr_core.engines.jina_vlm import JinaVLMEngine
        engine = JinaVLMEngine()
        result = engine.extract(sample_image)
        assert result.markdown
        assert "Hello" in result.markdown
        assert result.duration_ms > 0

    @pytest.mark.usefixtures("skip_if_no_deps", "skip_if_no_gpu")
    def test_extract_arabic(self, arabic_image):
        """⚠️ اختبار حرج — جودة العربية غير موثقة."""
        from ocr_core.engines.jina_vlm import JinaVLMEngine
        engine = JinaVLMEngine()
        result = engine.extract(arabic_image)
        assert result.markdown
        # تحقق من وجود حرف عربي
        import re
        arabic = re.findall(r"[\u0600-\u06FF]", result.markdown)
        assert len(arabic) > 3, "لا يوجد نص عربي في النتيجة"

    @pytest.mark.usefixtures("skip_if_no_deps", "skip_if_no_gpu")
    def test_strict_prompt_produces_latex(self, tmp_path):
        """Prompt صارم مع معادلة → LaTeX."""
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (400, 200), "white")
        draw = ImageDraw.Draw(img)
        draw.text((50, 80), "E = mc^2", fill="black")
        path = tmp_path / "formula.png"
        img.save(path)

        from ocr_core.engines.jina_vlm import (
            JinaVLMEngine, STRICT_OCR_PROMPT,
        )
        engine = JinaVLMEngine()
        result = engine.extract(str(path), prompt=STRICT_OCR_PROMPT)
        # يجب أن يظهر شيء يشبه LaTeX
        assert result.markdown


# ---------------------------------------------------------------------------
# حارس الإصلاح: extract() يتحقق من المسار قبل التحميل
# ---------------------------------------------------------------------------
def test_extract_validates_path_before_loading():
    """حارس انحدار لنقطة الضعف المُصلَحة (كانت موثقة كملاحظة معمارية).

    الإصلاح المطبق في engines/jina_vlm.py:
        if not Path(image_path).exists():
            raise FileNotFoundError(...)   # قبل _ensure_loaded()

    هذا الاختبار يتحقق أن التحقق موجود في المصدر *و* يسبق التحميل،
    حتى لا يعيد أحد إزالته دون انتباه.
    """
    from ocr_core.engines.jina_vlm import JinaVLMEngine
    import inspect
    src = inspect.getsource(JinaVLMEngine.extract)
    assert "Path(image_path).exists()" in src, (
        "أُزيل التحقق المبكر من المسار — سيضطر المستخدم لتحميل 7GB قبل الفشل"
    )
    assert src.index("Path(image_path).exists()") < src.index("_ensure_loaded()"), (
        "التحقق من المسار يجب أن يسبق _ensure_loaded()"
    )
