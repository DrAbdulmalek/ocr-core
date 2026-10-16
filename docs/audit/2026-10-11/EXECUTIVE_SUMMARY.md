# Executive summary — 2026-10-11

## What was actually changed
Three additive branches, three draft PRs. **No production branch was touched, nothing was merged, nothing deleted.**

| # | Repo | Branch | Commit | PR |
|---|---|---|---|---|
| 1 | ocr-core | `audit/evidence-scope-line-attribution` | `f7d2a8e` + corrective test commit | [#58](https://github.com/DrAbdulmalek/ocr-core/pull/58) draft |
| 2 | marathon_ted_pipeline | `audit/charter-engine-compat` | `c897389a5370809a5972726c9ee9eec663728732` | [#27](https://github.com/DrAbdulmalek/marathon_ted_pipeline/pull/27) draft |
| 3 | ocr-core | `audit/ecosystem-2026-10-11` | `77267822e90f` (+deliverables) | [#59](https://github.com/DrAbdulmalek/ocr-core/pull/59) draft |

## What was tested
Baseline on PR #57 head, then post-fix: focused 23 passed; full suite **515 passed, 7 skipped**.
Charter tripwire: 2 passed (offline, deterministic). Remote CI: re-read run IDs below.

## Ready for review
The evidence-scope fix (#58) once its corrective CI run is green — it is the only change that fixes a real defect.

## Unsafe to merge as-is
- PR #166 (**not** a draft, 45 open PRs in that repo, duplicated modules, no export ledger).
- PR #57 (leaves a third charter copy).
- Any flip of `require_visual_evidence` before a gated engine release exists.

## Requires owner approval
Drafting PR #166 · revoking the chat-exposed token · repairing the failing `Scheduled Tasks` and
`Build Android APK` workflows · canonicalising the duplicated modules · any dataset export.

## Next three highest-value actions
1. **Revoke the exposed PAT** (F-001) — one click, largest residual risk.
2. Merge a **gated ocr-core release** (take PR #57's `r16` + the #58 scope fix), then move the marathon pin and the boolean **together**.
3. Add the **export ledger + stored content hash** to the training path, and choose one canonical copy of the three duplicated modules.
