"""Synthetic tests for the offline model registry and archive importer."""
from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest

from ocr_core.model_store import ModelFile, ModelManifest, ModelRegistry, ModelStore


def _manifest_for(file_contents: dict[str, bytes], **overrides) -> ModelManifest:
    entries = tuple(
        ModelFile(path=name, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        for name, data in file_contents.items()
    )
    values = {
        "model_id": "example/tiny-ocr",
        "revision": "a" * 40,
        "task": "ocr",
        "framework": "onnx",
        "license_id": "MIT",
        "files": entries,
    }
    values.update(overrides)
    return ModelManifest(**values)


def _archive(path: Path, manifest: ModelManifest, files: dict[str, bytes], *, extra=None):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model-manifest.json", json.dumps(manifest.to_dict()))
        for name, data in files.items():
            zf.writestr(name, data)
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return path


def test_manifest_round_trip_is_canonical():
    manifest = _manifest_for({"weights/model.onnx": b"synthetic-weights"})
    restored = ModelManifest.from_dict(json.loads(manifest.canonical_bytes()))
    assert restored == manifest
    assert restored.storage_key == manifest.storage_key


@pytest.mark.parametrize("bad_path", ["../escape", "/absolute", "C:/drive", "a\\b", "a/./b", ""])
def test_manifest_rejects_unsafe_paths(bad_path):
    with pytest.raises(ValueError):
        ModelFile(path=bad_path, size_bytes=1, sha256="0" * 64)


@pytest.mark.parametrize("reserved", ["manifest.json", "model-manifest.json"])
def test_manifest_rejects_reserved_root_filenames(reserved):
    with pytest.raises(ValueError, match="reserved"):
        _manifest_for({reserved: b"not-allowed"})


def test_manifest_rejects_mutable_revision_names():
    with pytest.raises(ValueError, match="immutable"):
        _manifest_for({"weights.bin": b"x"}, revision="main")


def test_manifest_rejects_bad_digest_and_duplicate_paths():
    with pytest.raises(ValueError):
        ModelFile(path="weights.bin", size_bytes=1, sha256="not-a-hash")
    item = ModelFile(path="weights.bin", size_bytes=1, sha256=hashlib.sha256(b"x").hexdigest())
    with pytest.raises(ValueError, match="duplicate"):
        _manifest_for({"weights.bin": b"x"}, files=(item, item))


def test_registry_register_is_no_clobber_and_gets_by_identity(tmp_path):
    manifest = _manifest_for({"weights.bin": b"abc"})
    registry = ModelRegistry(tmp_path)
    first = registry.register(manifest)
    assert registry.register(manifest) == first
    assert registry.get(manifest.model_id, manifest.revision) == manifest
    other = _manifest_for({"weights.bin": b"xyz"})
    with pytest.raises(FileExistsError):
        registry.register(other)


def test_offline_import_verifies_and_registers_model(tmp_path):
    files = {"weights/model.onnx": b"synthetic-model-weights", "vocab.txt": b"abc"}
    manifest = _manifest_for(files)
    archive = _archive(tmp_path / "model.zip", manifest, files)
    store = ModelStore(tmp_path / "store")
    model_dir = store.import_archive(archive)
    assert store.verify(manifest)
    assert (model_dir / "weights/model.onnx").read_bytes() == files["weights/model.onnx"]
    assert store.get_verified(manifest.model_id, manifest.revision) == model_dir
    assert len(store.registry.list()) == 1


def test_import_rejects_hash_mismatch_and_leaves_no_registered_model(tmp_path):
    files = {"weights.bin": b"actual"}
    manifest = _manifest_for({"weights.bin": b"expect"})
    archive = _archive(tmp_path / "bad.zip", manifest, files)
    store = ModelStore(tmp_path / "store")
    with pytest.raises(ValueError, match="SHA-256"):
        store.import_archive(archive)
    assert store.registry.list() == []
    assert list(store.artifacts_dir.glob("*")) == []


def test_import_rejects_unlisted_files(tmp_path):
    files = {"weights.bin": b"abc"}
    manifest = _manifest_for(files)
    archive = _archive(tmp_path / "extra.zip", manifest, files, extra={"notes.txt": b"unexpected"})
    with pytest.raises(ValueError, match="exactly match"):
        ModelStore(tmp_path / "store").import_archive(archive)


def test_import_rejects_zip_slip_paths(tmp_path):
    files = {"../outside.txt": b"escape"}
    manifest = _manifest_for({"safe.bin": b"safe"})
    archive = tmp_path / "slip.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("model-manifest.json", json.dumps(manifest.to_dict()))
        zf.writestr("../outside.txt", b"escape")
    with pytest.raises(ValueError, match="unsafe|non-canonical"):
        ModelStore(tmp_path / "store").import_archive(archive)
    assert not (tmp_path / "outside.txt").exists()


def test_import_rejects_symlink_members(tmp_path):
    files = {"weights.bin": b"x"}
    manifest = _manifest_for(files)
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("weights.bin")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("model-manifest.json", json.dumps(manifest.to_dict()))
        zf.writestr(info, b"target")
    with pytest.raises(ValueError, match="symlink"):
        ModelStore(tmp_path / "store").import_archive(archive)


def test_import_enforces_member_and_total_size_limits(tmp_path):
    files = {"weights.bin": b"0123456789"}
    manifest = _manifest_for(files)
    archive = _archive(tmp_path / "large.zip", manifest, files)
    with pytest.raises(ValueError, match="size limit"):
        ModelStore(tmp_path / "store").import_archive(archive, max_file_size=5)


def test_store_verification_detects_post_import_tampering(tmp_path):
    files = {"weights.bin": b"original"}
    manifest = _manifest_for(files)
    archive = _archive(tmp_path / "good.zip", manifest, files)
    store = ModelStore(tmp_path / "store")
    model_dir = store.import_archive(archive)
    (model_dir / "weights.bin").write_bytes(b"tampered")
    assert store.verify(manifest) is False
    with pytest.raises(ValueError, match="integrity"):
        store.get_verified(manifest.model_id, manifest.revision)


def test_existing_verified_model_is_idempotent_but_corrupt_model_is_not_overwritten(tmp_path):
    files = {"weights.bin": b"weights"}
    manifest = _manifest_for(files)
    archive = _archive(tmp_path / "same.zip", manifest, files)
    store = ModelStore(tmp_path / "store")
    first = store.import_archive(archive)
    assert store.import_archive(archive) == first
    (first / "weights.bin").write_bytes(b"bad")
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        store.import_archive(archive)


def test_store_refuses_symlinked_root(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    files = {"weights.bin": b"safe"}
    manifest = _manifest_for(files)
    archive = _archive(tmp_path / "valid.zip", manifest, files)
    with pytest.raises(ValueError, match="symlink"):
        ModelStore(link).import_archive(archive)


def test_model_lease_transitions_state_and_is_exclusive(tmp_path):
    from ocr_core.model_store import ModelState

    files = {"weights.bin": b"leased"}
    manifest = _manifest_for(files)
    archive = _archive(tmp_path / "lease.zip", manifest, files)
    store = ModelStore(tmp_path / "store")
    store.import_archive(archive)
    assert store.status(manifest.model_id, manifest.revision) is ModelState.VERIFIED
    with store.lease(manifest.model_id, manifest.revision) as model_dir:
        assert model_dir == store.artifact_path(manifest)
        assert store.status(manifest.model_id, manifest.revision) is ModelState.IN_USE
        with pytest.raises(FileExistsError):
            with store.lease(manifest.model_id, manifest.revision):
                pass
    assert store.status(manifest.model_id, manifest.revision) is ModelState.VERIFIED


def test_stale_lease_is_reported_but_never_auto_removed(tmp_path):
    from ocr_core.model_store import ModelState

    files = {"weights.bin": b"stale"}
    manifest = _manifest_for(files)
    archive = _archive(tmp_path / "stale.zip", manifest, files)
    store = ModelStore(tmp_path / "store")
    store.import_archive(archive)
    leases = store.root / ".leases"
    leases.mkdir()
    lock = leases / (manifest.storage_key + ".lock")
    lock.write_text("999999\\n", encoding="ascii")
    assert store.status(manifest.model_id, manifest.revision) is ModelState.IN_USE
    with pytest.raises(FileExistsError):
        with store.lease(manifest.model_id, manifest.revision):
            pass
    assert lock.exists()


def test_get_verified_file_only_returns_manifest_declared_paths(tmp_path):
    files = {"weights/model.onnx": b"declared"}
    manifest = _manifest_for(files)
    archive = _archive(tmp_path / "file.zip", manifest, files)
    store = ModelStore(tmp_path / "store")
    store.import_archive(archive)
    assert store.get_verified_file(
        manifest.model_id, manifest.revision, "weights/model.onnx"
    ).read_bytes() == b"declared"
    with pytest.raises(KeyError):
        store.get_verified_file(manifest.model_id, manifest.revision, "other.bin")


def test_manifest_json_schema_is_packaged():
    from importlib.resources import files
    schema = json.loads(
        files("ocr_core").joinpath("config", "model_manifest.schema.yaml").read_text(encoding="utf-8")
    )
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    assert schema["properties"]["revision"]["pattern"] == "^[0-9a-f]{40,64}$"
    assert set(schema["required"]) == {
        "schema_version", "model_id", "revision", "task", "framework", "license_id", "files"
    }


def test_model_state_reports_missing_and_corrupt_artifacts(tmp_path):
    from ocr_core.model_store import ModelState

    files = {"weights.bin": b"state"}
    manifest = _manifest_for(files)
    store = ModelStore(tmp_path / "store")
    assert store.status(manifest.model_id, manifest.revision) is ModelState.NOT_REGISTERED
    store.registry.register(manifest)
    assert store.status(manifest.model_id, manifest.revision) is ModelState.MISSING
    archive = _archive(tmp_path / "state.zip", manifest, files)
    model_dir = store.import_archive(archive)
    assert store.status(manifest.model_id, manifest.revision) is ModelState.VERIFIED
    (model_dir / "weights.bin").write_bytes(b"tampered")
    assert store.status(manifest.model_id, manifest.revision) is ModelState.CORRUPT


def test_import_bounds_total_zip_entry_count(tmp_path):
    files = {"weights.bin": b"x"}
    manifest = _manifest_for(files)
    archive = tmp_path / "many-entries.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("model-manifest.json", json.dumps(manifest.to_dict()))
        zf.writestr("weights.bin", b"x")
        for index in range(5):
            zf.writestr(f"directory-{index}/", b"")
    with pytest.raises(ValueError, match="too many ZIP entries"):
        ModelStore(tmp_path / "store").import_archive(archive, max_entries=3)
