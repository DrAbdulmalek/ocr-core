# Dependency / API contract matrix (verified live)

| Consumer | Package it installs | Pin | Does the name match real metadata? | Gate support | Reproducible in CI | Evidence |
|---|---|---|---|---|---|---|
| `marathon_ted_pipeline` | `marathon-ocr-core` | `git+https://github.com/DrAbdulmalek/ocr-core.git@v0.6.0` (`requirements.txt:77`) | yes — `pyproject.toml [project] name = "marathon-ocr-core"` | **NO** — `require_visual_evidence` absent from `v0.6.0` engine | yes | tag-by-tag engine source inspection |
| `intelli-file-manager` | `marathon-ocr-core` (via `feat/harness-engineering`) | `PR #48` head `37920a6ef8ac` | yes | n/a | yes — [run 38177440179](https://github.com/DrAbdulmalek/intelli-file-manager/actions/runs/38177440179) success | live run re-read |
| `omni-medical-suite` PR #166 | **none** | — | — | — | — | `grep ocr_core` over all 8 changed Python files: 0 hits |

## Release-by-release gate capability (ocr-core)

| Tag | SHA | implements `require_visual_evidence`? |
|---|---|---|
| v0.1.0 | `6bf75a1a1c3b` | no |
| v0.2.0 | `76d586322a5f` | no |
| v0.3.0 | `1f5280ab7a90` | no |
| v0.4.0 | `20eeae7ab4a0` | no |
| v0.5.0 | `5613a1c05f7b` | no |
| v0.6.0 | `b0eff8387487` | no — `r16` returns only `{correct, wrong, ratio}` |

The gate exists only on the unmerged PR #57 branch (`e4e80e2370de`). **Conclusion: do not flip the marathon boolean yet; the compatible release does not exist.** Release order required: merge a gated engine release → then move the pin and the boolean together.

## Optional extras (ocr-core `pyproject.toml`, verified)
`tesseract`, `paddle`, `surya`, `easyocr`, `preprocess`, `benchmarks`, `golden`, `word-detect`, `jina`, `vlm`, `pdf`, `docx`, `all`, `dev`, `ui`, `ui-dev`.
`all` deliberately **excludes** `jina`/`surya`/`word-detect` (weight licences `CC BY-NC 4.0` / heavy deps) — this preserves the offline-first and licence contract.
