# Offline Model Registry and Store

## Purpose

This is a local-only foundation for model inventory and artifact integrity.
It does not download, resolve, execute, or activate a model. Engine integration
must be a separate reviewed change and must continue to obey
`ocr_core.model_policy` (LOCAL_ONLY by default).

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
