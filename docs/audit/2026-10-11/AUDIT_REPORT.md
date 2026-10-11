# Ecosystem audit — evidence-backed status, 2026-10-11

All SHAs, run IDs and conclusions below were re-read live from GitHub in this session.
See `pr_states.json`, `ci_evidence.json`, `repo_inventory.json|csv`, `findings_register.json`.

## 1. Inventory

`GET /user/repos?affiliation=owner,collaborator,organization_member` returned **55 repositories**
(32 public, 23 private) for `DrAbdulmalek` (id 101952860). **14 are archived.**
Machine-readable list: `repo_inventory.json` / `repo_inventory.csv`.

## 2. Verified PR states (live)

| PR | Title | Draft | Merged | Head SHA | Base SHA |
|---|---|---|---|---|---|
| ocr-core #57 | Enforce visual evidence before correctness/training labels | yes | no | `e4e80e2370de8a8ebf5fc3f85a9dd1d16f49ebda` | `main@e61683f7590d9f22c163af103a75f3c3e6db42b3` |
| omni-medical-suite #166 | Telegram OCR studio + pattern learning + training DB | **no** | no | `8077f5f1a362cea3d6622e51ffc7bcff611405f8` | `main@91dac6db3924667f3ba3f16189d816891cf3b34e` |
| marathon_ted_pipeline #26 | Agent-Reach medical/translation source catalog | yes | no | `ebf9df7b98dfb23f69e46b32f1d27ba411d4d98a` | `main@bcc0b57f7116a2731c52bac09e17a899639200f0` |
| intelli-file-manager #48 | Harness Engineering v2.0 | yes | no | `37920a6ef8ac71aed7ad3ac21d5b65761435eb3e` | `main@a6562a0380d5b58044b4ac8fdc444c053834b440` |

All reference SHAs in the assignment were still current. Open-PR load: ocr-core 5, marathon 16,
intelli-file-manager 11, omni-medical-suite **45**.

## 3. CI evidence (live conclusions)

| Workflow | Branch / SHA | Run | Conclusion |
|---|---|---|---|
| tests | ocr-core PR#57 `e4e80e2370de` | [38178234312](https://github.com/DrAbdulmalek/ocr-core/actions/runs/38178234312) | success |
| Secret Gate | ocr-core PR#57 `e4e80e2370de` | [38178234318](https://github.com/DrAbdulmalek/ocr-core/actions/runs/38178234318) | success |
| Python CI | intelli-file-manager `feat/harness-engineering` `37920a6ef8ac` | [38177440179](https://github.com/DrAbdulmalek/intelli-file-manager/actions/runs/38177440179) | success |
| Scheduled Tasks | marathon `main` `bcc0b57f7116` | [38137407655](https://github.com/DrAbdulmalek/marathon_ted_pipeline/actions/runs/38137407655) | **failure** (also 38129666592, 38041939315, 37916407378, 37762059884) |
| Build Android APK | intelli-file-manager `main` | 37690369977, 37683366703, 37599741264 | **failure** (all three) |
| 11 workflows | omni-medical-suite PR#166 `8077f5f1a362` | 38151419787 … 38151419703 | success |

## 4. Charter ↔ engine compatibility (re-verified)

- `marathon_ted_pipeline` `config/marathon_ocr_rules.yaml:243` → `require_visual_evidence: false`; `requirements.txt:77` → `marathon-ocr-core @ git+https://github.com/DrAbdulmalek/ocr-core.git@v0.6.0`.
- `ocr-core` PR#57 branch `src/ocr_core/config/marathon_ocr_rules.yaml:243` and `ocr_core/charter_v2.yaml:21` → `true`.
- **The pinned engine cannot implement the gate.** Inspecting every release tag: `v0.1.0 … v0.6.0` — `engine_hits=0` for `require_visual_evidence` in all six. `v0.6.0` `r16` returns only `{"correct","wrong","ratio"}` — pure ratio, no evidence concept, no `classification` key at all. The gate exists **only** on the unmerged PR #57 branch. Flipping the boolean in the marathon today would be inert.
- Trilogy of charter copies: marathon runtime config, `ocr-core/ocr_core/charter_v2.yaml` (declarative manifest), and `ocr-core/src/ocr_core/config/marathon_ocr_rules.yaml`. PR #57 turns the declarative copy into a pointer at the runtime file — the *right* direction, but it leaves PR #57's runtime config as a third copy that still drifts from the marathon's.

## 5. Reported defect: `branches: ain]` — CONTRADICTED

No such filter exists. Live reads:
`marathon/.github/workflows/ci.yml` → `branches: [main, develop]` (push), `[main]` (PR);
`omni-medical-dictionaries/.github/workflows/python-ci.yml` → `[ "main" ]`;
`dictionaries-csv/.github/workflows/python-ci.yml` → `[ "main" ]`.
The string `ain]` is produced by the code-search snippet truncating `[main]`. **No workflow was modified.**

## 6. Secret / privacy scan

- Pattern scan (`ghp_`, `github_pat_`, `AKIA`, `xox*`, `AIza`, PEM private keys) across the cloned trees and via `GET /search/code?q=ghp_+user:DrAbdulmalek` (total_count 0): **no committed credential found**.
- **The one real exposure is outside the repositories:** a live GitHub classic PAT was pasted into the chat transcript (finding F-001). It was never echoed, never written to a git remote, never committed; it existed only in a `chmod 600` sandbox file. **Owner action: revoke it.**
- `omni-medical-suite` PR#166 keeps Telegram credentials strictly in env / `$ZAI_SECRETS_DIR` (`app/services/telegram_ingest.py:32-59`) — no literal secret in the diff.
- `omni-ocr-training-db` is a **private** data store (patterns/slices/exports). Its 3 seed slices are Telegram **screenshot UI text** (`"Chats"`, `"Channels"`, `source.kind = telegram-screenshot`), i.e. scratch content, not clinical data — the privacy risk there is low, but the export path lacks an approval ledger (F-012).

## 7. Training-data lifecycle (as actually implemented)

Stages found in the PR#166 diff: correction capture (`slice_store`), pattern learning (`pattern_store`),
approval gate (`scripts/trainingdb_sync.py::validate_privacy_approvals`, per-sample, fail-closed),
deterministic split + export (`src/ocr/training_export.py`, sha256-based 80/10/10).
Stages **not** found: content-hash provenance in the stored records, an export ledger recording what/when/policy/source-revision, and explicit per-destination export authorization.
`omni-ocr-training-db` itself contains no schema or migration code.

## 8. Duplication

`omni-medical-suite` PR#166 adds each module twice — identical blobs:
`src/ocr/training_export.py` == `hf-space/src/ocr/training_export.py` (sha256 `0d0f45db142ea258`),
`pattern_store.py` (`a6e86f97c2312d0a`), `slice_store.py` (`d683f83d77a9452f`).
Same for F-013 (`ocr-core` root package shadow). Canonicalization proposed, not performed.

## 9. Offline-first

No cloud OCR/LLM fallback was introduced or found in the audited paths. `ocr-core` optional extras keep
heavy engines opt-in (`tesseract`, `paddle`, `easyocr`, `surya`, `jina`, `vlm`), and `all` deliberately
excludes `jina`/`surya` for weight-licence reasons (`pyproject.toml`).

## 10. Explicitly NOT done

Merging, drafting/un-drafting any PR, closing PRs, deleting/archiving repos or branches, publishing any dataset,
running Telegram ingestion/backfill, rotating the chat-exposed token, changing branch protection, or repairing
the failing `Scheduled Tasks` / `Build Android APK` workflows (both touch production paths).
