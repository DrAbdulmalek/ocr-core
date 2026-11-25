# OCR Core model access policy

OCR Core resolves optional model artifacts locally by default. The model
access policy mirrors the repository's `LOCAL_ONLY`, `EXPLICIT_REMOTE`,
and `REMOTE_DISABLED` modes without adding a network service or dependency.

- `LOCAL_ONLY` (default): use an already-cached artifact only. A cache miss
  is an error; the loader must not retry with remote access.
- `REMOTE_DISABLED`: use a local cache only, with remote authorization
  explicitly forbidden.
- `EXPLICIT_REMOTE`: remote artifact resolution is permitted only when the
  caller constructs `ModelAccessPolicy(mode="EXPLICIT_REMOTE",
  remote_authorized=True)` at an application boundary that has obtained
  explicit user authorization.

For Jina VLM, a local cache miss remains a clear error. For the automatic
word-detector choice, a missing local ONNX artifact falls back to the existing
ProjectionDetector; it does not fall back to a cloud service. Explicit
`kind="onnx"` remains fail-closed when the artifact is unavailable locally.

The policy controls model-file resolution only. It does not authorize sending
documents, images, DICOM, or extracted text to a remote inference provider.
No environment variable alone is treated as a security boundary.

**Supply-chain limitation:** Jina still requires `trust_remote_code=True` for
its current model implementation. Local-only resolution prevents fetching
model code at runtime, but does not audit code already present in the local
cache. Use only a separately reviewed and trusted cache.
