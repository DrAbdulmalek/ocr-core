"""Engine adapters resolve only verified local ModelStore artifacts."""
from pathlib import Path

import pytest


def test_onnx_detector_uses_verified_store_without_huggingface(monkeypatch, tmp_path):
    from ocr_core.preprocess.word_detector import OnnxWordDetector
    expected = tmp_path / "verified" / "model.onnx"
    calls = []

    class StoreStub:
        def __init__(self, root):
            calls.append(("root", Path(root)))

        def get_verified_file(self, model_id, revision, relative_path):
            calls.append((model_id, revision, relative_path))
            return expected

    monkeypatch.setattr("ocr_core.model_store.ModelStore", StoreStub)
    detector = OnnxWordDetector(model_store_root=tmp_path / "store", model_revision="a" * 40)
    assert detector._resolve_model() == expected
    assert calls[-1] == (detector.model_id, "a" * 40, "model.onnx")


def test_onnx_store_requires_revision(tmp_path):
    from ocr_core.preprocess.word_detector import OnnxWordDetector
    with pytest.raises(ValueError, match="model_revision"):
        OnnxWordDetector(model_store_root=tmp_path)


def test_jina_resolves_verified_snapshot_directory(monkeypatch, tmp_path):
    from ocr_core.engines.jina_vlm import JinaVLMEngine
    expected = tmp_path / "verified" / "snapshot"

    class StoreStub:
        def __init__(self, root):
            assert Path(root) == tmp_path / "store"

        def get_verified(self, model_id, revision):
            assert model_id == "jinaai/jina-ocr-v1"
            assert revision == "b" * 40
            return expected

    monkeypatch.setattr("ocr_core.model_store.ModelStore", StoreStub)
    engine = JinaVLMEngine(model_store_root=tmp_path / "store", model_revision="b" * 40)
    assert engine._resolve_model_source() == str(expected)


def test_jina_store_requires_revision(tmp_path):
    from ocr_core.engines.jina_vlm import JinaVLMEngine
    with pytest.raises(ValueError, match="model_revision"):
        JinaVLMEngine(model_store_root=tmp_path)
