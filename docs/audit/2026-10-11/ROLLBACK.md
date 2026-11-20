# Rollback guide — 2026-10-11 audit writes

Every write in this audit is on a dedicated `audit/*` branch. **No production branch was touched.**

| Write | Repo | Branch | Commit | Rollback |
|---|---|---|---|---|
| Evidence-scope line attribution fix | ocr-core | `audit/evidence-scope-line-attribution` | see PR | `git revert <sha>` on the branch, or delete the branch (nothing on `main` changed) |
| Charter↔engine compatibility tripwire | marathon_ted_pipeline | `audit/charter-engine-compat` | `c897389a5370809a5972726c9ee9eec663728732` | `git revert c897389` on the branch |
| Audit artifacts (this folder) | ocr-core | `audit/ecosystem-2026-10-11` | see PR | delete the branch |

Each branch is additive and based on the current `main` SHA recorded in the PR body. No force-push, no history rewrite, no branch/tag deletion, no repository or release deletion was performed.
