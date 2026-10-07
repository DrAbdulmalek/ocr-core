# Production Layout Analyzer Boundary and Hypothesis Quality Contract

## Purpose
Define the boundary for a production medical layout analyzer before introducing any ML/model implementation. The analyzer is an observation/proposal component; it does not own the canonical LayoutDocument and does not perform matching, review, or apply.

## 1. Input boundary
A production analyzer consumes source-image pixel evidence plus explicit source/document identity and analyzer configuration. Canonical coordinates remain source-image pixels: origin top-left, +X right, +Y down, bbox `(x,y,width,height)`, nonnegative dimensions, and no implicit DPI conversion, clipping, or rounding. Other coordinate spaces require explicit versioned conversion.

## 2. Output boundary
The analyzer emits LayoutAnalysisResult. Each result carries analyzer name/version, configuration ID, source ID, and analysis-run ID when available. RegionHypothesis identity is analysis-scoped and must never be treated as persistent semantic identity. The analyzer must not assign MedicalRegionData.region_id as canonical authority.

## 3. Hypothesis validity
Every hypothesis must have unique identity within the result, finite bbox coordinates, nonnegative width/height, canonical coordinate compliance, deterministic serialization, and preserved provenance. Out-of-image geometry is not silently clipped.

## 4. Quality metadata
Analyzer quality metadata is observation metadata, not OCR confidence. Scores require explicit scale, meaning, producer, version, and calibration policy. Quality never authorizes matching or apply.

## 5. Provenance
Provenance distinguishes analyzer identity, analyzer version, configuration identity, source/document identity, and analysis-run identity. Model/configuration changes create distinct analysis provenance and never alter persistent region identity.

## 6. Empty and degraded results
Zero hypotheses means only that this run produced no hypotheses; it does not delete existing regions. Partial, degraded, or failed analysis must be represented explicitly where distinguishable and must never silently replace the canonical document.

## 7. Determinism
For identical evidence, provenance, configuration, and analyzer version, ordering, serialization, and hypothesis identity generation should be deterministic. Nondeterminism must be explicitly documented. Determinism is not semantic identity.

## 8. Failure boundary
Analyzer failure must not mutate LayoutDocument, delete regions, clear manual state, create persistent identities, invoke matching, or invoke apply.

## 9. Baseline compatibility
The deterministic projection analyzer is the reference implementation of the current contract. Future production analyzers must remain compatible with LayoutAnalysisResult, RegionHypothesis, provenance, coordinate rules, deterministic serialization, and non-destructive reconciliation.

## 10. Separation of concerns
Required flow: source evidence → analyzer → LayoutAnalysisResult → explicit cross-run matching → review/decision → explicit apply → LayoutDocument. Analyzer must not perform semantic matching, modify canonical regions, approve hypotheses, create/delete persistent regions, execute OCR, or publish datasets.

## 11. Evolution
Model/configuration changes must be visible in provenance and must not silently reuse run identity, mutate persistent identity, or authorize application.

## 12. Auditability
Serialized results should establish analyzer/version/configuration, source, analysis run, and emitted hypotheses. Quality metadata remains distinct from matching evidence and OCR confidence.

## 13. Versioning
The analyzer contract is independent of layout_version, OCR confidence calibration, matching policy version, and lifecycle metadata version. Contract changes require explicit versioning and compatibility review.

## 14. Required runtime gate
Tests must cover: valid pixel input; invalid/non-rectangular input; finite/nonnegative geometry; deterministic repeated execution; unique hypothesis IDs; analysis-run provenance; empty-result semantics; degraded/failure non-mutation; configuration/model provenance; no persistent-ID creation; no implicit matching/apply; serialization round-trip; baseline compatibility; canonical/manual document non-mutation.

## 15. Non-goals
No ML model selection, OCR, semantic matching, automatic apply, deletion/deactivation policy, or SQLite/HF/React integration.

## Safety invariant
Analyzer output = observation/proposal.
Hypothesis identity != persistent region identity.
Quality score != OCR confidence != authorization.
Empty result != deletion.
Analyzer failure != document mutation.
Analysis != matching != apply.
