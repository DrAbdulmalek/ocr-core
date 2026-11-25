# Offline Model Registry and Store

## Purpose

This is a local-only foundation for model inventory and artifact integrity.
It does not download, resolve, execute, or activate a model. Engine integration
must be a separate reviewed change and remain offline by default with explicit
authorization for any remote access. The proposed `ocr_core.model_policy`
implementation is tracked separately in PR #81 and is not included here.

## Manifest format

The packaged schema is `ocr_core/config/model_manifest.schema.yaml`. A ZIP import contains one root `model-manifest.json` and exactly the files
declared by that manifest. Example:

```json
{
  "schema_version": "1.0",
  "model_id": "example/tiny-ocr",
  "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "task": "ocr",
  "framework": "onnx",
  "license_id": "MIT",
  "files": [
    {
      "path": "weights/model.onnx",
      "size_bytes": 123456,
      "sha256": "64-lowercase-hex-characters"
    }
  ],
  "metadata": {
    "language": ["ar", "en"]
  }
}
```

Every file path is a canonical relative POSIX path. Absolute paths, drive
prefixes, backslashes, dot segments, parent traversal, duplicate paths,
symlinks, special files, encrypted entries, unexpected files, and oversized
archives are rejected. SHA-256 and exact byte size are verified while
streaming into a private staging directory.

## Storage and lifecycle

- Registry entries are keyed by SHA-256 of model ID + NUL + revision.
- Artifact directories use the same opaque key; untrusted identifiers never
  become filesystem paths.
- Import publishes only after all files pass size and digest verification.
- Existing verified artifacts are idempotent; corrupt artifacts are never
  overwritten automatically.
- `ModelStore.verify()` is read-only and checks the exact file set, sizes,
  hashes, and stored manifest.
- `ModelStore.get_verified()` refuses missing or modified artifacts.

## Lifecycle and concurrent use

The read-only state API reports `NOT_REGISTERED`, `MISSING`, `CORRUPT`,
`VERIFIED`, or `IN_USE`. A lease is acquired with
`with store.lease(model_id, revision) as model_dir:`; the lock is created
exclusively, so a second process cannot acquire the same model concurrently.
The lease is removed on normal exit. If a process crashes, its lock remains
and the state stays `IN_USE`; this is deliberately fail-closed. Inspect and
resolve stale locks manually only after verifying that the owning process is
gone. The store never guesses and never deletes a stale lock automatically.

Use `get_verified_file()` to resolve a file only if it is declared by the
manifest and the complete artifact passes integrity verification.

## Explicit non-goals

No network access, automatic model downloads, model execution, cache eviction,
garbage collection, or medical accuracy claims. License identifiers and
metadata are recorded, not legally validated. A human must review each model's
license, provenance, intended use, and resource requirements before activation.


## Engine integration

ModelStore can be supplied to OnnxWordDetector and JinaVLMEngine with model_store_root and an immutable model_revision (40–64 lowercase hex characters). ONNX additionally accepts model_file (default model.onnx). The engines resolve only through get_verified_file() or get_verified(); integrity failures are surfaced and never trigger a Hugging Face retry. The Jina snapshot must contain the complete local Transformers processor/model files. No model weights are bundled with this repository.


## Engine lease ownership

When an engine is configured with `model_store_root`, it holds an exclusive
ModelStore lease for as long as its ONNX session or Jina model/processor is
loaded. Call the engine's `close()` method when the engine is no longer in
use; this releases the lease and clears loaded engine references. If model
loading fails, the engine releases the lease before propagating the error.
