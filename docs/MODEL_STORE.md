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


## Jina custom-code trust boundary

Jina custom model code is executable Python. The engine therefore defaults
`trust_remote_code=False`, including when a snapshot has passed ModelStore
hash verification. Set `trust_remote_code=True` only after independently
reviewing the exact model code and accepting its execution. A matching SHA-256
proves byte integrity relative to the manifest; it does not prove that code
is safe. ModelStore does not audit Python code.


## Local engine smoke test

After importing a model archive and reviewing its license/provenance, run the
smoke-test helper on the machine that will host OCR. It creates a blank
synthetic image, prints only timings/counts, and never prints recognized text
or accesses a network source:

```bash
python scripts/smoke_test_model_store.py \\
  --store /path/to/model-store \\
  --model-id example/model \\
  --revision <40-to-64-char-commit-sha> \\
  --engine onnx --model-file weights/model.onnx
```

Jina is optional and resource-intensive. Its weights are marked CC BY-NC 4.0
in this repository; review the license and available memory before use. Jina
custom Python code remains disabled unless explicitly opted into after code
review:

```bash
python scripts/smoke_test_model_store.py --store /path/to/model-store \\
  --model-id jinaai/jina-ocr-v1 --revision <commit-sha> --engine jina \\
  --device cpu --trust-remote-code
```

The helper does not establish OCR accuracy: a blank synthetic image validates
only model resolution, loading, and one inference call. It is not a clinical
benchmark and must not be run on patient documents as part of this smoke test.


## Model-specific license and capability notes (source review)

The following are upstream model-card facts, not an endorsement for clinical use:

- `PellelNitram/xournalpp-htr-word-detector`: model card lists MIT, ONNX, 448×448 grayscale input, and describes the output as word bounding boxes only (not transcription). It reports training/validation on IAM-style handwriting and explicitly warns that other writing styles may differ. It is therefore a detector component, not an Arabic medical handwriting recognizer.
- `jinaai/jina-ocr-v1`: model card lists CC BY-NC 4.0, describes a 3B-parameter model, and its Transformers example requires `trust_remote_code=True`. Commercial use is not granted by that license. This project defaults custom code execution to false and requires an explicit caller opt-in after code review.

Upstream cards:
- https://huggingface.co/PellelNitram/xournalpp-htr-word-detector
- https://huggingface.co/jinaai/jina-ocr-v1

Neither card establishes accuracy on Arabic medical handwriting or Syrian clinical documents. Local evaluation on an appropriately authorized, privacy-preserving dataset remains mandatory before clinical use.
