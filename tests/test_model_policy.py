"""اختبارات سياسة الوصول للنماذج — محلي-فقط افتراضيًا (OCR-05/#79).

كلها اصطناعية وبلا شبكة وبلا نماذج ثقيلة: stubs عبر sys.modules تحاكي
huggingface_hub/transformers/onnxruntime. تغطي قائمة التحقق في القضية:
استيراد بلا HF ولا شبكة، التصريح الصريح، سلوك نقص الخزنة، والـfallback
المحلي للمسار الآلي، وفشل الاختيار الصريح مغلقًا.
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from ocr_core.model_policy import (
    ENV_AUTHORIZED,
    ENV_POLICY,
    EXPLICIT_REMOTE,
    LOCAL_ONLY,
    REMOTE_DISABLED,
    ModelAccessPolicy,
    ModelArtifactMissingError,
)


# ─── 1. الاستيراد بلا HF ولا شبكة ────────────────────────────────────────────

class TestNoNetworkImports:
    def test_core_imports_no_hf_no_network(self):
        """استيراد وحدات النواة لا يستورد huggingface_hub/transformers/torch."""
        # جذر الاستيراد الوحيد = src/ (منع أي تظليل) — محقون في الكود لأن
        # الوضع المعزول -I يتجاهل PYTHONPATH (بلا اعتماد على تثبيت editable)
        src_root = str(Path(__file__).resolve().parents[1] / "src")
        code = (
            f"import sys; sys.path.insert(0, {src_root!r}); "
            "import ocr_core.model_policy, ocr_core.preprocess.word_detector, "
            "ocr_core.engines.jina_vlm; "
            "banned = ('huggingface_hub', 'transformers', 'torch', 'onnxruntime', 'requests'); "
            "leaked = [m for m in banned if m in sys.modules]; "
            "assert not leaked, f'وحدات شبكية تسربت عند الاستيراد: {leaked}'; "
            "print('CLEAN')"
        )
        out = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr
        assert "CLEAN" in out.stdout

    def test_policy_module_is_stdlib_only(self):
        """وحدة السياسة نفسها بلا أي استيراد ثقيل عند الاستدعاء."""
        policy = ModelAccessPolicy.resolve()
        assert policy.allows_remote() is False


# ─── 2. التصريح الصريح ───────────────────────────────────────────────────────

class TestExplicitAuthorization:
    def test_default_is_local_only(self):
        assert ModelAccessPolicy.resolve().mode == LOCAL_ONLY
        assert ModelAccessPolicy.resolve().allows_remote() is False

    def test_explicit_remote_requires_authorization(self):
        p = ModelAccessPolicy.resolve(EXPLICIT_REMOTE, False)
        assert p.allows_remote() is False
        p = ModelAccessPolicy.resolve(EXPLICIT_REMOTE, True)
        assert p.allows_remote() is True

    def test_remote_disabled_beats_authorization(self):
        p = ModelAccessPolicy.resolve(REMOTE_DISABLED, True)
        assert p.allows_remote() is False

    def test_env_precedence_and_override(self, monkeypatch):
        monkeypatch.setenv(ENV_POLICY, EXPLICIT_REMOTE)
        monkeypatch.setenv(ENV_AUTHORIZED, "1")
        assert ModelAccessPolicy.resolve().allows_remote() is True
        # البارامتر الصريح يتفوق على البيئة
        assert ModelAccessPolicy.resolve(LOCAL_ONLY, None).allows_remote() is False

    def test_unknown_mode_fails_closed(self):
        with pytest.raises(ValueError):
            ModelAccessPolicy.resolve("SOMETING_ELSE", True)


# ─── 3. سلوك نقص الخزنة (stubs بلا شبكة) ─────────────────────────────────────

class _HfStub(types.ModuleType):
    """يحاكي huggingface_hub.hf_hub_download ويسجل نداءاته."""

    calls: list = []

    @staticmethod
    def hf_hub_download(repo_id, filename, cache_dir=None, local_files_only=False):
        _HfStub.calls.append({"local_files_only": local_files_only})
        if local_files_only:
            raise OSError("Local entry not found (اصطناعي — خزنة فارغة)")
        raise OSError("لا شبكة في الاختبار — لن يُنفَّذ البُعد فعليًا")


@pytest.fixture
def hf_stub(monkeypatch):
    _HfStub.calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", _HfStub("huggingface_hub"))
    return _HfStub


class TestOnnxCacheMiss:
    def test_local_only_raises_without_remote_retry(self, hf_stub, tmp_path):
        """LOCAL_ONLY + خزنة فارغة → فشل مغلق، صفر نداء بعيد."""
        from ocr_core.preprocess.word_detector import OnnxWordDetector

        det = OnnxWordDetector(cache_dir=tmp_path)
        with pytest.raises(ModelArtifactMissingError):
            det._resolve_model()
        assert len(hf_stub.calls) == 1  # محاولة محلية واحدة فقط
        assert hf_stub.calls[0]["local_files_only"] is True

    def test_explicit_remote_attempts_one_remote_call(self, hf_stub, tmp_path):
        """EXPLICIT_REMOTE مصرّح → نداء بعيد واحد صريح (يمنع التكرار بالتصميم)."""
        from ocr_core.preprocess.word_detector import OnnxWordDetector

        det = OnnxWordDetector(cache_dir=tmp_path, policy=EXPLICIT_REMOTE,
                               remote_authorized=True)
        with pytest.raises(OSError, match="لا شبكة في الاختبار"):
            det._resolve_model()
        assert [c["local_files_only"] for c in hf_stub.calls] == [True, False]

    def test_has_local_artifact_false_on_empty_cache(self, hf_stub, tmp_path):
        from ocr_core.preprocess.word_detector import OnnxWordDetector

        det = OnnxWordDetector(cache_dir=tmp_path)
        assert det.has_local_artifact() is False


# ─── 4. الـfallback المحلي للمسار الآلي ──────────────────────────────────────

class _OrtStub(types.ModuleType):
    """يحاكي وجود onnxruntime لتشغيل فرع auto الغني بالأثر."""


class TestAutoPathLocalFallback:
    def test_auto_falls_back_when_artifact_missing(self, hf_stub, monkeypatch):
        """auto + onnxruntime متاح + أثر غائب + LOCAL_ONLY → ProjectionDetector."""
        monkeypatch.setitem(sys.modules, "onnxruntime", _OrtStub("onnxruntime"))
        from ocr_core.preprocess.word_detector import create_detector

        det = create_detector("auto")
        assert det.name == "projection"

    def test_auto_keeps_onnx_when_remote_authorized(self, hf_stub, monkeypatch):
        """auto + EXPLICIT_REMOTE مصرّح → يبقى onnx (التنزيل عند أول استخدام)."""
        monkeypatch.setitem(sys.modules, "onnxruntime", _OrtStub("onnxruntime"))
        from ocr_core.preprocess.word_detector import create_detector

        det = create_detector("auto", policy=EXPLICIT_REMOTE, remote_authorized=True)
        assert det.name == "onnx"

    def test_explicit_onnx_remains_fail_closed(self, hf_stub, tmp_path):
        """الاختيار الصريح kind='onnx' + أثر غائب → فشل مغلق عند التحميل."""
        from ocr_core.preprocess.word_detector import create_detector

        det = create_detector("onnx", cache_dir=tmp_path)
        assert det.name == "onnx"
        with pytest.raises(ModelArtifactMissingError):
            det._resolve_model()


# ─── 5. jina_vlm محلي أولًا (stubs بلا torch حقيقي) ──────────────────────────

def _install_light_stubs(monkeypatch, fail_local: bool = True):
    """يحاكي torch + transformers للتحميل المحلي بلا أوزان حقيقية."""
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _TinyModule:
        def to(self, device):
            return self

    transformers_stub = types.ModuleType("transformers")

    class _Proc:
        @staticmethod
        def from_pretrained(model_id, local_files_only=False, **kw):
            if local_files_only:
                raise OSError("Local entry not found (اصطناعي)")
            raise OSError("لا شبكة في الاختبار")

    class _Model(_Proc):
        @staticmethod
        def from_pretrained(model_id, local_files_only=False, **kw):
            return _TinyModule()

    transformers_stub.AutoProcessor = _Proc
    transformers_stub.AutoModelForCausalLM = _Model
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)


class TestJinaLocalOnly:
    def test_jina_local_only_fails_closed(self, monkeypatch):
        """LOCAL_ONLY + خزنة فارغة → ModelArtifactMissingError بلا شبكة."""
        _install_light_stubs(monkeypatch)
        from ocr_core.engines.jina_vlm import JinaVLMEngine

        eng = JinaVLMEngine(device="cpu")
        with pytest.raises(ModelArtifactMissingError):
            eng._ensure_loaded()

    def test_jina_remote_authorized_reaches_remote_call(self, monkeypatch):
        """EXPLICIT_REMOTE مصرّح → يمرّ للمحاولة البعدية (stub يرفض بلا شبكة)."""
        _install_light_stubs(monkeypatch)
        from ocr_core.engines.jina_vlm import JinaVLMEngine

        eng = JinaVLMEngine(device="cpu", policy=EXPLICIT_REMOTE, remote_authorized=True)
        with pytest.raises(OSError, match="لا شبكة في الاختبار"):
            eng._ensure_loaded()
