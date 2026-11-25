# OCR Pipeline Provenance

The pipeline preserves engine metadata through the existing `meta` mappings;
it does not add fields to the locked `OCRResult` dataclass or alter the
OCRResult JSON contract.

For a full-page recognition, `PageResult.meta["recognition"]` records the
engine name, confidence value, whether that value is known (zero means
unknown), processing time, page count, and a shallow copy of `OCRResult.meta`.

For region-wise recognition, the same structure is attached to each
`RegionRecord.meta["recognition"]`. The aggregate page recognition metadata
also contains a region list with each region's bbox and provenance.

The page metadata additionally records the inferred reading direction and
region count. Raw recognized text remains in `PageResult.text`; normalization
and corrections remain separate. Metadata values are copied, not interpreted
as proof of model identity or license. A consumer must validate any model
digest or provenance claims against a verified local ModelStore manifest.

No real patient data is required for these tests; fixtures use synthetic text
and synthetic model identifiers only.
