# Chandra OCR adapter (optional)

This adapter invokes the documented Datalab Chandra CLI as a local subprocess and reads its Markdown output. It does not import PyTorch or Chandra into the core process, bundle model weights, call a hosted API, or silently switch to a remote service.

## Install

Install the adapter's CLI dependency:

```bash
pip install -e ".[chandra]"
```

For local Hugging Face inference, install Chandra's heavyweight HF dependencies in the same environment:

```bash
pip install "chandra-ocr[hf]>=0.2.0"
```

Chandra documents two inference methods:
- `hf`: local Hugging Face inference.
- `vllm`: expects a separately configured vLLM server. Verify that its endpoint is local or explicitly controlled; this adapter does not configure or launch a server.

The adapter passes `--method` explicitly and defaults to `hf` to avoid an implicit server/backend choice. This is not a promise of acceptable CPU performance: Chandra is a large vision-language OCR model, and CPU-only inference on a 16 GB laptop may be impractically slow or memory-constrained. Validate in a separate environment before using it on routine workloads.

## Usage

```python
from ocr_core.engines.chandra import ChandraEngine

engine = ChandraEngine(method="hf", timeout=900)
result = engine.process_pdf("document.pdf", max_pages=3)
if result.error:
    print("Chandra failed:", result.error)
else:
    print(result.text)
```

Outputs are Markdown, so structural markers are intentionally preserved. Confidence is always `0.0` (unknown); no confidence is fabricated. For PDFs, the page count remains unknown (0) until the adapter parses a verified metadata schema. The temporary output directory is removed after reading the result.

## Privacy, clinical safety, and licensing

- The adapter runs a local CLI. It does not upload documents itself, but a user-configured vLLM endpoint may be remote; verify the endpoint before processing confidential material.
- Never send identifiable patient documents to an unapproved remote endpoint. Use synthetic/de-identified test fixtures and keep patient data out of Git, CI logs, and benchmark artifacts.
- The repository's code is Apache-2.0, but Chandra's model weights are under a modified OpenRAIL-M license with additional use restrictions. Review the current upstream `MODEL_LICENSE` and Datalab commercial-usage terms before downloading, redistributing weights, offering a service, or using it in a commercial clinical workflow.
- Chandra output must not independently determine medication doses, negation, diagnosis, or other clinical decisions. Apply the medical review and evidence/provenance gates downstream.

## Validation status

This adapter and its tests are a proposal. Mocked tests validate command construction and result handling only; they do not prove that a real Chandra model can load, run on CPU, read Arabic medical records, or meet a performance target. Those require an isolated smoke test and a versioned Arabic/English medical golden set with CER/WER and layout checks.
