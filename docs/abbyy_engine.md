# ABBYY FineReader 16 as an optional private OCR source

## Privacy and licence boundary

The project owner reports a valid ABBYY FineReader 16 licence and permission to use relevant weights for private experiments, provided the weights and experimental outputs are not published and experiment results are shared with the rights holder.

This is treated as private-use scope only. Do not commit, upload, publish, attach to public issues/PRs, or place in public releases any ABBYY weights, extracted model data, licensed binaries, APK contents, proprietary vocabularies, private scans, or experimental outputs that must remain private. Keep such material on local disk or explicitly private storage with access controls verified before use.

The ocr-core repository is public. This PR therefore contains only a generic adapter, fake-process tests, and documentation. It does not contain licensed weights or proprietary model artifacts. The three research repositories remain research references rather than runtime dependencies. Do not port experimental reverse-engineered model reconstruction into runtime until the permitted scope is verified.

## Runtime adapter

AbbyyFineReaderEngine calls a user-configured local bridge. It does not guess a vendor-specific CLI and does not itself implement FineReader 16's Windows GUI/API. The bridge must use an installation or API covered by the user's licence.

Configuration is read from local environment variables; do not commit machine paths or licensed material:
- OCR_CORE_ABBYY_COMMAND: JSON array of arguments, not a shell command string.
- OCR_CORE_ABBYY_OUTPUT_MODE: file (default) or stdout.
- OCR_CORE_ABBYY_TIMEOUT: timeout seconds (default 120).

Example for a user-created documented bridge; these are illustrative flags, not official ABBYY command-line flags:

```bash
export OCR_CORE_ABBYY_COMMAND='["/usr/local/bin/my-licensed-abbyy-bridge","--input","{input}","--output-text","{output}","--language","Arabic,English"]'
export OCR_CORE_ABBYY_OUTPUT_MODE=file
export OCR_CORE_ABBYY_TIMEOUT=120
```

```python
from ocr_core.engines.abbyy import AbbyyFineReaderEngine
engine = AbbyyFineReaderEngine.from_env()
if not engine.available():
    raise RuntimeError("ABBYY bridge is not configured or unavailable")
result = engine.process_image("scan.png")
if result.error:
    raise RuntimeError(result.error)
print(result.text)
```

The command is passed as an argument vector with shell=False and a timeout. A genuine confidence score is never fabricated: confidence=0.0 means unknown. File-mode output uses a temporary directory.

## Linux / Manjaro and FineReader 16

FineReader 16 is a Windows application; this adapter does not make it natively Linux-compatible. Evaluate these options in order:
1. A documented ABBYY SDK/API covered by the licence and callable from Linux.
2. A local Windows bridge/VM that invokes the licensed installation and returns OCR text/results. Keep it bound to localhost or a private interface, authenticate cross-host requests, and never expose it to the public internet.
3. A separate Wine-prefix experiment without assuming compatibility. Installation alone does not prove OCR automation, licensing, or file-format support. Do not bypass activation/licensing controls.

The adapter cannot guarantee a third-party bridge stays offline. If private-use conditions require local processing, verify network behavior and apply OS/network controls to block outbound access as appropriate. Do not send medical data to cloud services without explicit authorization.

## Combining ABBYY with open-source engines

Keep ABBYY optional and run it beside installed engines such as Tesseract, PaddleOCR, and EasyOCR. Do not automatically trust the longest output. Preserve raw outputs and provenance, then compare using the same evaluation set and normalization.

For Arabic handwriting and medical text, evaluate CER/WER, Arabic letter/diacritic preservation and RTL order, medical terms, numbers, units, laterality, negation, region order, latency, memory, failures, and reproducibility.

Do not promote an output to gold/training data from model agreement alone. Require human correction and explicit approval, preserve original image/text, and record engine/version/configuration plus dataset/result hashes. Keep private experimental data out of public fixtures, CI logs, and benchmark artifacts.

## What is and is not proven

Unit tests use a fake process. They do not prove FineReader 16 or Wine compatibility, licence scope, weight access, Arabic quality, or the existence of a working bridge. Before enabling ABBYY in default routing: confirm the permitted workflow, prove local OCR on a representative scan, benchmark against open-source engines, review CER/WER and medical-critical errors, and keep the engine opt-in until validation passes.

No router default is changed by this adapter-only PR.