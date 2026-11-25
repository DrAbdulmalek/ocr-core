"""Offline model manifests, integrity verification, and safe ZIP import.

This module is deliberately stdlib-only. It never downloads models and never
extracts arbitrary archive paths. Model files become visible only after every
declared size and SHA-256 digest has been verified in a staging directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

_MANIFEST_NAME = "model-manifest.json"
_STORED_MANIFEST_NAME = "manifest.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def _safe_relative_path(value: str) -> PurePosixPath:
    """Validate a portable archive-relative POSIX path."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("model file path must be a non-empty string")
    if "\\" in value or value.startswith("/") or _DRIVE_RE.match(value):
        raise ValueError(f"unsafe model file path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or not path.parts:
        raise ValueError(f"non-canonical model file path: {value!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe model file path: {value!r}")
    return path


@dataclass(frozen=True)
class ModelFile:
    """One required model artifact with an exact size and SHA-256 digest."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.path)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ValueError("size_bytes must be a positive integer")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class ModelManifest:
    """Immutable, reproducible description of a model revision and its files."""

    model_id: str
    revision: str
    task: str
    framework: str
    license_id: str
    files: tuple[ModelFile, ...]
    schema_version: str = "1.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("model_id", "revision", "task", "framework", "license_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported model manifest schema: {self.schema_version!r}")
        if not self.files:
            raise ValueError("manifest must declare at least one model file")
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest contains duplicate file paths")
        reserved = {_MANIFEST_NAME, _STORED_MANIFEST_NAME}
        if reserved.intersection(paths):
            raise ValueError("manifest uses a reserved root metadata filename")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "task": self.task,
            "framework": self.framework,
            "license_id": self.license_id,
            "files": [item.to_dict() for item in sorted(self.files, key=lambda item: item.path)],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelManifest":
        if not isinstance(value, Mapping):
            raise ValueError("manifest must be a JSON object")
        allowed = {"schema_version", "model_id", "revision", "task", "framework", "license_id", "files", "metadata"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        raw_files = value.get("files")
        if not isinstance(raw_files, list):
            raise ValueError("manifest files must be a list")
        files = []
        for raw in raw_files:
            if not isinstance(raw, Mapping) or set(raw) != {"path", "size_bytes", "sha256"}:
                raise ValueError("each manifest file must contain exactly path, size_bytes, sha256")
            files.append(ModelFile(**raw))
        return cls(
            model_id=value.get("model_id"),
            revision=value.get("revision"),
            task=value.get("task"),
            framework=value.get("framework"),
            license_id=value.get("license_id"),
            files=tuple(files),
            schema_version=value.get("schema_version", "1.0"),
            metadata=value.get("metadata", {}),
        )

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @property
    def storage_key(self) -> str:
        identity = (self.model_id + "\0" + self.revision).encode("utf-8")
        return hashlib.sha256(identity).hexdigest()


class ModelState(str, Enum):
    """Conservative local lifecycle state derived from registry, integrity, and lease."""

    NOT_REGISTERED = "NOT_REGISTERED"
    MISSING = "MISSING"
    CORRUPT = "CORRUPT"
    VERIFIED = "VERIFIED"
    IN_USE = "IN_USE"


class ModelRegistry:
    """Local manifest registry. Registration is atomic and no-clobber by default."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifests_dir = self.root / "manifests"

    def _prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("model registry root must not be a symlink")
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        if self.manifests_dir.is_symlink():
            raise ValueError("model registry manifests directory must not be a symlink")

    def register(self, manifest: ModelManifest, *, overwrite: bool = False) -> Path:
        self._prepare()
        target = self.manifests_dir / (manifest.storage_key + ".json")
        payload = manifest.canonical_bytes()
        if target.exists():
            if target.is_symlink():
                raise ValueError("refusing symlinked manifest entry")
            if target.read_bytes() == payload:
                return target
            if not overwrite:
                raise FileExistsError(f"a different manifest is already registered for {manifest.model_id}@{manifest.revision}")
        fd, temp_name = tempfile.mkstemp(prefix=".manifest-", dir=self.manifests_dir)
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temp, target)
            else:
                # Hard-link publication is atomic and fails rather than clobbering
                # an entry created concurrently by another importer.
                os.link(temp, target)
            return target
        finally:
            temp.unlink(missing_ok=True)

    def get(self, model_id: str, revision: str) -> ModelManifest:
        key = hashlib.sha256((model_id + "\0" + revision).encode("utf-8")).hexdigest()
        path = self.manifests_dir / (key + ".json")
        if path.is_symlink():
            raise ValueError("refusing symlinked manifest entry")
        return ModelManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[ModelManifest]:
        self._prepare()
        result = []
        for path in sorted(self.manifests_dir.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"unsafe registry entry: {path.name}")
            result.append(ModelManifest.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return result


class ModelStore:
    """Verified local model store with bounded, traversal-safe ZIP import."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.staging_dir = self.root / ".staging"
        self.registry = ModelRegistry(self.root)

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def artifact_path(self, manifest: ModelManifest) -> Path:
        return self.artifacts_dir / manifest.storage_key

    def verify(self, manifest: ModelManifest) -> bool:
        """Verify exact file set, sizes, hashes, and stored manifest; no mutation."""
        root = self.artifact_path(manifest)
        if root.is_symlink() or not root.is_dir():
            return False
        expected = {item.path: item for item in manifest.files}
        actual: set[str] = set()
        for current, dirs, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            for dirname in dirs:
                if (current_path / dirname).is_symlink():
                    return False
            for filename in names:
                path = current_path / filename
                if path.is_symlink() or not path.is_file():
                    return False
                relative = path.relative_to(root).as_posix()
                if relative == _STORED_MANIFEST_NAME:
                    continue
                actual.add(relative)
                spec = expected.get(relative)
                if spec is None or path.stat().st_size != spec.size_bytes:
                    return False
                if self._digest(path) != spec.sha256:
                    return False
        if actual != set(expected):
            return False
        stored_manifest = root / _STORED_MANIFEST_NAME
        if stored_manifest.is_symlink() or not stored_manifest.is_file():
            return False
        try:
            stored = ModelManifest.from_dict(json.loads(stored_manifest.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        return stored.canonical_bytes() == manifest.canonical_bytes()

    def import_archive(
        self,
        archive_path: str | Path,
        *,
        max_files: int = 256,
        max_total_size: int = 2 * 1024 * 1024 * 1024,
        max_file_size: int = 1024 * 1024 * 1024,
        max_compression_ratio: float = 200.0,
        max_manifest_size: int = 1024 * 1024,
    ) -> Path:
        """Import a local ZIP only after strict validation; never accesses network."""
        archive = Path(archive_path)
        if archive.is_symlink() or not archive.is_file():
            raise ValueError("archive must be an existing regular file, not a symlink")
        if max_files < 1 or max_total_size < 1 or max_file_size < 1 or max_compression_ratio < 1:
            raise ValueError("archive limits must be positive")

        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("model store root must not be a symlink")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        if self.artifacts_dir.is_symlink() or self.staging_dir.is_symlink():
            raise ValueError("model store directories must not be symlinks")

        stage: Path | None = None
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                files: dict[str, zipfile.ZipInfo] = {}
                total_size = 0
                for info in zf.infolist():
                    raw_name = info.filename[:-1] if info.is_dir() and info.filename.endswith("/") else info.filename
                    relative = _safe_relative_path(raw_name)
                    mode = info.external_attr >> 16
                    file_type = stat.S_IFMT(mode)
                    if file_type == stat.S_IFLNK:
                        raise ValueError(f"symlink entry is forbidden: {info.filename!r}")
                    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise ValueError(f"special archive entry is forbidden: {info.filename!r}")
                    if info.is_dir() and file_type not in (0, stat.S_IFDIR):
                        raise ValueError(f"directory entry has a non-directory type: {info.filename!r}")
                    if not info.is_dir() and file_type == stat.S_IFDIR:
                        raise ValueError(f"file entry has a directory type: {info.filename!r}")
                    if info.is_dir():
                        continue
                    if info.flag_bits & 0x1:
                        raise ValueError(f"encrypted archive entry is forbidden: {info.filename!r}")
                    name = relative.as_posix()
                    if name in files:
                        raise ValueError(f"duplicate archive entry: {name!r}")
                    if info.file_size > max_file_size:
                        raise ValueError(f"archive member exceeds size limit: {name!r}")
                    ratio = info.file_size / max(info.compress_size, 1)
                    if ratio > max_compression_ratio:
                        raise ValueError(f"archive member compression ratio exceeds limit: {name!r}")
                    total_size += info.file_size
                    if total_size > max_total_size:
                        raise ValueError("archive exceeds total uncompressed size limit")
                    files[name] = info
                    if len(files) > max_files + 1:  # one extra entry is the manifest
                        raise ValueError("archive contains too many files")

                manifest_info = files.get(_MANIFEST_NAME)
                if manifest_info is None or manifest_info.file_size > max_manifest_size:
                    raise ValueError("archive must contain a bounded root model-manifest.json")
                try:
                    manifest_data = json.loads(zf.read(manifest_info).decode("utf-8"))
                    manifest = ModelManifest.from_dict(manifest_data)
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"invalid model manifest in archive: {exc}") from exc

                expected = {item.path: item for item in manifest.files}
                actual = set(files) - {_MANIFEST_NAME}
                if actual != set(expected):
                    raise ValueError("archive file set does not exactly match the manifest")
                for item in manifest.files:
                    if files[item.path].file_size != item.size_bytes:
                        raise ValueError(f"declared size does not match archive member: {item.path}")

                stage = Path(tempfile.mkdtemp(prefix="import-", dir=self.staging_dir))
                for item in manifest.files:
                    info = files[item.path]
                    relative = _safe_relative_path(item.path)
                    destination = stage.joinpath(*relative.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    written = 0
                    with zf.open(info, "r") as source, destination.open("xb") as target:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            written += len(chunk)
                            if written > item.size_bytes or written > max_file_size:
                                raise ValueError(f"archive member expanded beyond declared size: {item.path}")
                            digest.update(chunk)
                            target.write(chunk)
                    if written != item.size_bytes:
                        raise ValueError(f"actual size does not match manifest: {item.path}")
                    if digest.hexdigest() != item.sha256:
                        raise ValueError(f"SHA-256 mismatch for model file: {item.path}")

                (stage / _STORED_MANIFEST_NAME).write_bytes(manifest.canonical_bytes())
                target_dir = self.artifact_path(manifest)
                if target_dir.exists():
                    if self.verify(manifest):
                        shutil.rmtree(stage)
                        stage = None
                        self.registry.register(manifest)
                        return target_dir
                    raise FileExistsError("model artifact directory exists but does not verify; refusing overwrite")
                # Atomic same-filesystem directory publication; concurrent non-empty
                # destination directories cause rename to fail rather than overwrite.
                os.rename(stage, target_dir)
                stage = None
                self.registry.register(manifest)
                return target_dir
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)

    def status(self, model_id: str, revision: str) -> ModelState:
        """Return a read-only lifecycle state; stale leases are never auto-cleared."""
        try:
            manifest = self.registry.get(model_id, revision)
        except FileNotFoundError:
            return ModelState.NOT_REGISTERED
        artifact = self.artifact_path(manifest)
        if not artifact.exists():
            return ModelState.MISSING
        if not self.verify(manifest):
            return ModelState.CORRUPT
        lock = self.root / ".leases" / (manifest.storage_key + ".lock")
        if lock.is_symlink():
            raise ValueError("refusing symlinked model lease")
        if lock.exists():
            return ModelState.IN_USE
        return ModelState.VERIFIED

    @contextmanager
    def lease(self, model_id: str, revision: str) -> Iterator[Path]:
        """Hold an exclusive process-safe lease while a verified model is in use.

        A process crash leaves a stale lock intentionally. The store never
        guesses whether a process is dead and never removes stale locks itself.
        """
        manifest = self.registry.get(model_id, revision)
        model_dir = self.get_verified(model_id, revision)
        leases_dir = self.root / ".leases"
        leases_dir.mkdir(parents=True, exist_ok=True)
        if leases_dir.is_symlink():
            raise ValueError("model leases directory must not be a symlink")
        lock = leases_dir / (manifest.storage_key + ".lock")
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        lock_stat = os.fstat(fd)
        try:
            os.write(fd, (str(os.getpid()) + "\n").encode("ascii"))
            os.fsync(fd)
            yield model_dir
        finally:
            os.close(fd)
            try:
                current = lock.lstat()
                if (current.st_dev, current.st_ino) == (lock_stat.st_dev, lock_stat.st_ino):
                    lock.unlink()
            except FileNotFoundError:
                pass

    def get_verified_file(self, model_id: str, revision: str, relative_path: str) -> Path:
        """Resolve one declared file only after the complete model verifies."""
        relative = _safe_relative_path(relative_path)
        manifest = self.registry.get(model_id, revision)
        if relative_path not in {item.path for item in manifest.files}:
            raise KeyError(f"file is not declared by the model manifest: {relative_path}")
        model_dir = self.get_verified(model_id, revision)
        return model_dir.joinpath(*relative.parts)


__all__ = ["ModelFile", "ModelManifest", "ModelRegistry", "ModelState", "ModelStore"]
