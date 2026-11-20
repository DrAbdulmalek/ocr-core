# Source catalog — status of the Agent-Reach catalog (PR #26)

`marathon_ted_pipeline` PR #26 adds `docs/source-discovery/agent-reach-medical-translation-source-catalog.md`
(92 lines, **documentation only** — 1 file changed). Head `ebf9df7b98df`, draft, open, CI green
([run 38174496046](https://github.com/DrAbdulmalek/marathon_ted_pipeline/actions/runs/38174496046)).

## Verification status — evidence-based

| Claim | Status |
|---|---|
| The catalog exists in the PR | PROVEN (file list re-read) |
| The catalog is verified documentation | PROVEN |
| **Agent-Reach is installed** | **UNPROVEN — not claimed** |
| **Agent-Reach is operational / adds measurable value** | **UNPROVEN — experiment not run in this session** |
| Any source listed is approved for redistribution or ML training | **UNPROVEN — the catalog is a discovery list, not a licence determination** |

No source in the catalog may be labelled "approved", "installed" or "operational" on the strength of
the PR alone. Each entry still needs: canonical URL, owner, licence/terms, redistribution permission,
commercial-use permission, ML-training permission, date checked, and verification evidence.

## Isolation rules applied (and to apply for any future Agent-Reach evaluation)
Public/official medical sources only (WHO / WHO EMRO, NCBI / PubMed, Europe PMC, EMA, regional indexes);
verify each one's actual terms rather than assuming public accessibility implies a redistribution right;
run Agent-Reach only in a temporary environment with synthetic or public test inputs; never send secrets
or patient data; never install it globally or make it a mandatory runtime dependency; never touch live
Telegram source lists, target channels, credentials, ingestion cursors or production config; no marathon
run and no historical backfill — dry-run and synthetic fixtures only.

**Verdict: production unchanged. Integration not proposed.**
