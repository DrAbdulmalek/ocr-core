# ABBYY FineReader as an optional OCR source

## Integration boundary

The ABBYY research repositories are treated as **research references**, not as
runtime dependencies:

- [abbyy-e2e-research](https://github.com/DrAbdulmalek/abbyy-e2e-research)
  documents static findings and explicitly says that original model files and
  extracted weights are not included.
- [abbyy-neoml-toolkit](https://github.com/DrAbdulmalek/abbyy-neoml-toolkit)
  contains experimental parsers, vocabularies, architecture summaries, and
  model-reconstruction work. Its own README requires a license decision,
  security review, and CER benchmark before production use.
- [finereader-ocr-apk-analysis PR #1](https://github.com/DrAbdulmalek/finereader-ocr-apk-analysis/pull/1)
  is a draft static-analysis documentation change; the repository's default
  branch is currently empty. Its observations are not runtime capabilities.

This integration does **not** copy ABBYY binaries, APKs, proprietary model
files, extracted weights, or reverse-engineered model reconstruction code into
`ocr-core`. It does not bypass licensing or cloud/API access controls.

## Runtime adapter

`ocr_core.engines.abbyy.AbbyyFineReaderEngine` is a dependency-free adapter for
a command-line bridge supplied by the user. The bridge must call a locally
installed ABBYY product/SDK that the user is licensed to use, or another
authorized integration layer. The adapter intentionally does not guess a
vendor-specific command syntax because it varies by product, platform, and
version.

The command is an argument list, not a shell string. Use these placeholders:

- `{input}` — absolute path to the input image.
- `{output}` — temporary text-output path (required in `output_mode="file"`).

Example with a **user-provided local bridge** (the command shown is illustrative,
not an ABBYY vendor command):

```python
from ocr_core.engines.abbyy import AbbyyFineReaderEngine

engine = AbbyyFineReaderEngine(
    [
        "/usr/local/bin/my-licensed-abbyy-bridge",
        "--input", "{input}",
        "--output-text", "{output}",
        "--language", "Arabic,English",
    ],
    output_mode="file",
    timeout=120,
)
if engine.available():
    result = engine.process_image("scan.png")
    if result.error:
        print("ABBYY source unavailable:", result.error)
    else:
        print(result.text)
```

For a bridge that writes recognized text to stdout:

```python
engine = AbbyyFineReaderEngine(
    ["/path/to/authorized-bridge", "--input", "{input}"],
    output_mode="stdout",
)
```

Do not put shell operators, secrets, or credentials in command strings. The
adapter invokes `subprocess` with `shell=False`, applies a timeout, captures
errors, and reports failure through `OCRResult.error`. It never fabricates a
confidence value: `confidence=0.0` means unknown.

## What is and is not proven

The unit tests use a fake local command to validate the adapter contract. They
do not prove that ABBYY FineReader is installed, that a specific ABBYY edition
supports a particular CLI, or that any ABBYY model has been successfully
loaded. Those facts must be established on the target machine using the
authorized installation.

Before enabling this engine in a default routing or ensemble profile:

1. Confirm the exact ABBYY product/edition and license permits the intended use.
2. Implement or configure a bridge matching that product's documented API.
3. Test Arabic and English on a licensed, representative benchmark set.
4. Record CER/WER, RTL behavior, runtime, failures, and provenance.
5. Compare against existing `ocr-core` engines with identical normalization
   (`normalize_v1`) and avoid normalizing twice.
6. Keep it opt-in until the benchmark and security review pass.

No engine routing default is changed by this adapter-only integration.
