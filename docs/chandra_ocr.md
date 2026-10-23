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

Outputs are Markdown, so structural markers are intentionally preserved. Confidence is always `0.0` (unknown); no confidence is fabricated.

Failure semantics (verified live against chandra-ocr 0.2.0): the CLI exits 0 and writes an EMPTY `<stem>.md` even when inference fails (for example, an unreachable vLLM backend, after internal retries). The adapter therefore treats empty or whitespace-only Markdown output as an explicit failure, never as a successful blank page.

Page counts for PDFs are read from the sibling `<stem>_metadata.json` when that file exists and is valid (schema verified against chandra-ocr 0.2.0: `num_pages`, `total_token_count`, `total_chunks`, per-page list). When the metadata is missing or unreadable the page count stays unknown (0) and nothing is fabricated; when the metadata records `total_token_count: 0`, the result carries a `generation_evidence: zero_tokens_recorded` flag for downstream diagnosis. The temporary output directory is removed after reading the result.

## Privacy, clinical safety, and licensing

- The adapter runs a local CLI. It does not upload documents itself, but a user-configured vLLM endpoint may be remote; verify the endpoint before processing confidential material.
- Never send identifiable patient documents to an unapproved remote endpoint. Use synthetic/de-identified test fixtures and keep patient data out of Git, CI logs, and benchmark artifacts.
- The repository's code is Apache-2.0, but Chandra's model weights are under a modified OpenRAIL-M license with additional use restrictions. Review the current upstream `MODEL_LICENSE` and Datalab commercial-usage terms before downloading, redistributing weights, offering a service, or using it in a commercial clinical workflow.
- Chandra output must not independently determine medication doses, negation, diagnosis, or other clinical decisions. Apply the medical review and evidence/provenance gates downstream.

## Validation status

This adapter and its tests are a proposal. Mocked tests validate command construction and result handling only; they do not prove that a real Chandra model can load, run on CPU, read Arabic medical records, or meet a performance target. Those require an isolated smoke test and a versioned Arabic/English medical golden set with CER/WER and layout checks.

Verified so far (2026-10-23, chandra-ocr 0.2.0 base install, no model weights): CLI argument contract (`--method`, `--page-range`), output layout (`<output>/<stem>/<stem>.md` plus sibling `<stem>_metadata.json`), hf-backend crash surfacing (non-zero exit is reported), and the empty-output failure semantics above, reproduced by a dedicated regression test before and after the fix. Not yet verified: real-model inference on this hardware (weights are ~10.6 GB), Arabic medical OCR quality (CER/WER on a medical golden set), and the vLLM path against a live server.
