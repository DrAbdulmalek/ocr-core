"""Engine adapters resolve only verified local ModelStore artifacts."""
from contextlib import contextmanager
from pathlib import Path

import pytest


def test_onnx_detector_uses_verified_store_without_huggingface(monkeypatch, tmp_path):
    from ocr_core.preprocess.word_detector import OnnxWordDetector
    expected = tmp_path / "verified" / "model.onnx"
    calls = []

    class StoreStub:
        def __init__(self, root):
            calls.append(("root", Path(root)))

        @contextmanager
        def lease(self, model_id, revision):
            calls.append(("lease-enter", model_id, revision))
            try:
                yield expected.parent
            finally:
                calls.append(("lease-exit", model_id, revision))

        def get_verified_file(self, model_id, revision, relative_path):
            calls.append((model_id, revision, relative_path))
            return expected

    monkeypatch.setattr("ocr_core.model_store.ModelStore", StoreStub)
    detector = OnnxWordDetector(model_store_root=tmp_path / "store", model_revision="a" * 40)
    assert detector._resolve_model() == expected
    assert calls[-1] == (detector.model_id, "a" * 40, "model.onnx")
    assert ("lease-enter", detector.model_id, "a" * 40) in calls
    detector.close()
    assert ("lease-exit", detector.model_id, "a" * 40) in calls


def test_onnx_store_requires_revision(tmp_path):
    from ocr_core.preprocess.word_detector import OnnxWordDetector
    with pytest.raises(ValueError, match="model_revision"):
        OnnxWordDetector(model_store_root=tmp_path)


def test_jina_resolves_verified_snapshot_directory(monkeypatch, tmp_path):
    from ocr_core.engines.jina_vlm import JinaVLMEngine
    expected = tmp_path / "verified" / "snapshot"
    seen = {}

    class StoreStub:
        def __init__(self, root):
            assert Path(root) == tmp_path / "store"

        @contextmanager
        def lease(self, model_id, revision):
            assert model_id == "jinaai/jina-ocr-v1"
            assert revision == "b" * 40
            try:
                yield expected
            finally:
                seen["lease_released"] = True

    monkeypatch.setattr("ocr_core.model_store.ModelStore", StoreStub)
    engine = JinaVLMEngine(model_store_root=tmp_path / "store", model_revision="b" * 40)
    assert engine._resolve_model_source() == str(expected)
    assert "lease_released" not in seen
    engine.close()
    assert seen["lease_released"] is True


def test_jina_store_requires_revision(tmp_path):
    from ocr_core.engines.jina_vlm import JinaVLMEngine
    with pytest.raises(ValueError, match="model_revision"):
        JinaVLMEngine(model_store_root=tmp_path)


def test_onnx_engine_loads_from_verified_store_path(monkeypatch, tmp_path):
    import sys
    import types
    from ocr_core.preprocess.word_detector import OnnxWordDetector

    model_path = tmp_path / "verified" / "weights.onnx"
    seen = {}
    class StoreStub:
        def __init__(self, root): pass
        @contextmanager
        def lease(self, model_id, revision):
            seen["lease_entered"] = True
            try:
                yield model_path.parent
            finally:
                seen["lease_released"] = True
        def get_verified_file(self, model_id, revision, relative_path):
            seen["store"] = (model_id, revision, relative_path)
            return model_path
    class Options:
        graph_optimization_level = None
        intra_op_num_threads = 0
    class Runtime(types.ModuleType):
        SessionOptions = Options
        GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=1)
        @staticmethod
        def InferenceSession(path, **kwargs):
            seen["session"] = (path, kwargs)
            return object()
    monkeypatch.setattr("ocr_core.model_store.ModelStore", StoreStub)
    monkeypatch.setitem(sys.modules, "onnxruntime", Runtime("onnxruntime"))
    detector = OnnxWordDetector(
        model_store_root=tmp_path / "store",
        model_revision="c" * 40,
        model_file="weights.onnx",
    )
    detector._ensure_loaded()
    assert detector._loaded is True
    assert seen["session"][0] == str(model_path)
    assert seen["store"][2] == "weights.onnx"
    assert seen["lease_entered"] is True
    assert "lease_released" not in seen
    detector.close()
    assert seen["lease_released"] is True


def test_jina_engine_loads_verified_snapshot_without_remote_resolution(monkeypatch, tmp_path):
    import sys
    import types
    from ocr_core.engines.jina_vlm import JinaVLMEngine

    snapshot = tmp_path / "verified" / "jina"
    seen = {"processor": [], "model": []}
    class StoreStub:
        def __init__(self, root): pass
        @contextmanager
        def lease(self, model_id, revision):
            seen["lease_entered"] = True
            try:
                yield snapshot
            finally:
                seen["lease_released"] = True
    class Model:
        def to(self, device): return self
        def eval(self): pass
    class Processor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            seen["processor"].append((source, kwargs))
            return object()
    class AutoModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            seen["model"].append((source, kwargs))
            return Model()
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.bfloat16 = object()
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = Processor
    transformers.AutoModelForCausalLM = AutoModel
    monkeypatch.setattr("ocr_core.model_store.ModelStore", StoreStub)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    engine = JinaVLMEngine(
        model_store_root=tmp_path / "store",
        model_revision="d" * 40,
    )
    engine._ensure_loaded()
    assert engine._model is not None
    assert seen["processor"][0][0] == str(snapshot)
    assert seen["model"][0][0] == str(snapshot)
    assert seen["processor"][0][1]["local_files_only"] is True
    assert seen["model"][0][1]["local_files_only"] is True
    assert seen["lease_entered"] is True
    assert "lease_released" not in seen
    engine.close()
    assert seen["lease_released"] is True
