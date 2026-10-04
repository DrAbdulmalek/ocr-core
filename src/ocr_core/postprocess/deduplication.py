"""Field-aware deduplication and semantic search helpers for medical OCR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from rapidfuzz import fuzz

from ocr_core.postprocess.field_extractor import ArabicMedicalFieldExtractor, ExtractedMedicalFields
from ocr_core.postprocess.normalization import arabic_strong_normalize

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore[assignment]
    Distance = PointStruct = VectorParams = None  # type: ignore[assignment]

DEFAULT_FIELD_WEIGHTS = {
    "patient_name": 0.35,
    "patient_id": 0.30,
    "date": 0.15,
    "diagnosis": 0.10,
    "medications": 0.05,
    "template_signature": 0.05,
}


@dataclass(slots=True)
class SimilarityResult:
    score: float
    field_scores: dict[str, float]
    is_same_patient: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class WeightedMedicalDeduplicator:
    """Deduplicate records using patient-aware field weighting."""

    def __init__(
        self,
        extractor: ArabicMedicalFieldExtractor | None = None,
        weights: dict[str, float] | None = None,
        duplicate_threshold: float = 0.85,
    ) -> None:
        self.extractor = extractor or ArabicMedicalFieldExtractor()
        self.weights = weights or DEFAULT_FIELD_WEIGHTS.copy()
        self.duplicate_threshold = duplicate_threshold

    @staticmethod
    def _ratio(left: str, right: str) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return fuzz.ratio(arabic_strong_normalize(left), arabic_strong_normalize(right)) / 100.0

    def _coerce(self, record: str | dict[str, Any] | ExtractedMedicalFields) -> ExtractedMedicalFields:
        if isinstance(record, ExtractedMedicalFields):
            return record
        if isinstance(record, dict):
            if "raw_text" in record or "patient_name" in record:
                return ExtractedMedicalFields(
                    patient_name=str(record.get("patient_name", "")),
                    patient_id=str(record.get("patient_id", "")),
                    date=str(record.get("date", "")),
                    doctor_name=str(record.get("doctor_name", "")),
                    diagnosis=str(record.get("diagnosis", "")),
                    medications=list(record.get("medications", []) or []),
                    template_signature=str(record.get("template_signature", "")),
                    raw_text=str(record.get("raw_text", record.get("text", ""))),
                )
            return self.extractor.extract_fields(str(record.get("text", "")))
        return self.extractor.extract_fields(record)

    def compare(
        self,
        left: str | dict[str, Any] | ExtractedMedicalFields,
        right: str | dict[str, Any] | ExtractedMedicalFields,
    ) -> SimilarityResult:
        left_fields = self._coerce(left)
        right_fields = self._coerce(right)

        field_scores = {
            "patient_name": self._ratio(left_fields.patient_name, right_fields.patient_name),
            "patient_id": self._ratio(left_fields.patient_id, right_fields.patient_id),
            "date": self._ratio(left_fields.date, right_fields.date),
            "diagnosis": self._ratio(left_fields.diagnosis, right_fields.diagnosis),
            "medications": self._ratio(" | ".join(left_fields.medications), " | ".join(right_fields.medications)),
            "template_signature": self._ratio(left_fields.template_signature, right_fields.template_signature),
        }

        weighted_score = 0.0
        total_weight = 0.0
        for field_name, weight in self.weights.items():
            left_value = getattr(left_fields, field_name, "")
            right_value = getattr(right_fields, field_name, "")
            if left_value or right_value:
                weighted_score += field_scores[field_name] * weight
                total_weight += weight

        if total_weight == 0:
            total_weight = 1.0
            weighted_score = self._ratio(left_fields.raw_text, right_fields.raw_text)

        final_score = weighted_score / total_weight
        critical_match = (
            field_scores["patient_name"] >= 0.92
            or field_scores["patient_id"] >= 0.98
        )
        is_same_patient = final_score >= self.duplicate_threshold and critical_match

        if not is_same_patient and field_scores["template_signature"] >= 0.9:
            explanation = "Template is highly similar but patient-identifying fields diverge."
        elif is_same_patient:
            explanation = "High weighted match on patient-identifying fields."
        else:
            explanation = "Low weighted similarity across patient-critical fields."

        return SimilarityResult(
            score=round(final_score, 4),
            field_scores={k: round(v, 4) for k, v in field_scores.items()},
            is_same_patient=is_same_patient,
            explanation=explanation,
        )

    def deduplicate(self, records: list[str | dict[str, Any] | ExtractedMedicalFields]) -> dict[str, Any]:
        unique: list[ExtractedMedicalFields] = []
        duplicates: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            current = self._coerce(record)
            matched = False
            for unique_index, candidate in enumerate(unique):
                similarity = self.compare(current, candidate)
                if similarity.is_same_patient:
                    duplicates.append(
                        {
                            "source_index": index,
                            "matched_unique_index": unique_index,
                            "similarity": similarity.to_dict(),
                            "record": current.to_dict(),
                        }
                    )
                    matched = True
                    break
            if not matched:
                unique.append(current)

        # P1-3: instrument decision
        try:
            from app.core.decision_log import log_decision

            log_decision(
                decision="dedup_batch",
                outcome={
                    "input_count": len(records),
                    "unique_count": len(unique),
                    "duplicate_count": len(duplicates),
                },
                reasons=[
                    f"duplicate_threshold={self.duplicate_threshold}",
                    f"weights={self.weights}",
                ],
                inputs={
                    "input_count": len(records),
                    "unique_count": len(unique),
                },
                duration_ms=None,
            )
        except ImportError:
            pass  # app.core.decision_log not available

        return {
            "unique_records": [record.to_dict() for record in unique],
            "duplicates": duplicates,
            "input_count": len(records),
            "unique_count": len(unique),
        }


def field_aware_similarity(
    left: str | dict[str, Any] | ExtractedMedicalFields,
    right: str | dict[str, Any] | ExtractedMedicalFields,
    extractor: ArabicMedicalFieldExtractor | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    deduplicator = WeightedMedicalDeduplicator(extractor=extractor, weights=weights)
    return deduplicator.compare(left, right).to_dict()


class QdrantMedicalSearch:
    """Semantic search with Qdrant when available, fuzzy fallback otherwise."""

    def __init__(
        self,
        qdrant_url: str | None = None,
        collection_name: str = "omni_medical_suite_records",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        extractor: ArabicMedicalFieldExtractor | None = None,
    ) -> None:
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.model_name = model_name
        self.extractor = extractor or ArabicMedicalFieldExtractor()
        self._records: list[dict[str, Any]] = []
        self._model = SentenceTransformer(model_name) if SentenceTransformer is not None else None
        self._client = QdrantClient(url=qdrant_url) if (qdrant_url and QdrantClient is not None) else None

    def _payload_text(self, record: dict[str, Any]) -> str:
        meds = " ".join(record.get("medications", []) or [])
        return "\n".join(
            part
            for part in [
                record.get("patient_name", ""),
                record.get("patient_id", ""),
                record.get("date", ""),
                record.get("diagnosis", ""),
                meds,
                record.get("raw_text", record.get("text", "")),
            ]
            if part
        )

    def _ensure_record(self, record: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(record, dict) and ("raw_text" in record or "patient_name" in record):
            ensured = dict(record)
            ensured.setdefault("raw_text", record.get("text", ""))
            return ensured
        if isinstance(record, dict):
            text = str(record.get("text", ""))
        else:
            text = record
        return self.extractor.extract_fields(text).to_dict()

    def upsert_records(self, records: list[str | dict[str, Any]]) -> dict[str, Any]:
        prepared = [self._ensure_record(record) for record in records]
        self._records = prepared
        if not self._client or not self._model or not VectorParams or not Distance or not PointStruct:
            return {"backend": "local", "indexed": len(prepared)}

        vectors = self._model.encode([self._payload_text(record) for record in prepared]).tolist()
        vector_size = len(vectors[0]) if vectors else 0
        self._client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=index, vector=vector, payload=record)
            for index, (vector, record) in enumerate(zip(vectors, prepared, strict=False))
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)
        return {"backend": "qdrant", "indexed": len(prepared)}

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        if self._client and self._model:
            embedding = self._model.encode([query])[0].tolist()
            hits = self._client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=top_k,
            )
            return [
                {
                    "id": str(hit.id),
                    "score": round(float(hit.score), 4),
                    "text": self._payload_text(hit.payload),
                    "metadata": hit.payload,
                    "backend": "qdrant",
                }
                for hit in hits
            ]

        query_norm = arabic_strong_normalize(query)
        scored: list[SearchHit] = []
        for index, record in enumerate(self._records):
            haystack = self._payload_text(record)
            fuzzy = fuzz.token_set_ratio(query_norm, arabic_strong_normalize(haystack)) / 100.0
            diagnosis_boost = 0.05 if query_norm and query_norm in arabic_strong_normalize(record.get("diagnosis", "")) else 0.0
            score = min(1.0, fuzzy + diagnosis_boost)
            if score > 0:
                scored.append(SearchHit(id=str(index), score=score, text=haystack, metadata=record))
        scored.sort(key=lambda item: item.score, reverse=True)
        return [
            {
                "id": hit.id,
                "score": round(hit.score, 4),
                "text": hit.text,
                "metadata": hit.metadata,
                "backend": "local-fuzzy",
            }
            for hit in scored[:top_k]
        ]
