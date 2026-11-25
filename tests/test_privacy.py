"""اختبارات ``ocr_core.privacy`` — استعادة وحدة pii-scrubbing (OCR-04/#76).

كل البيانات هنا اصطناعية خيالية بالكامل — قاعدة الوحدة نفسها تحظر إدخال
بيانات مرضى حقيقية في git. تغطي: طبقة Regex (هواتف/تواريخ/بريد/هويات)،
حماية جرعات الأدوية، الدفعات، سجلات jsonl، فحص البقايا، تعطيل NER
الرشيق بلا شبكة، وثوابت السياسة.
"""
from __future__ import annotations

import sys

import pytest

from ocr_core.privacy import (
    PRIVACY_POLICY,
    HybridPIIScrubber,
    RedactionStats,
    scrub_pii,
)


# ─── طبقة Regex: الهواتف ─────────────────────────────────────────────────────

class TestRegexPhones:
    def setup_method(self):
        self.scrubber = HybridPIIScrubber(use_ner=False)

    def test_syrian_phone(self):
        redacted, stats = self.scrubber.scrub("الهاتف: +963-911-234-567")
        assert "[REDACTED_PHONE]" in redacted
        assert stats.phones_found > 0

    def test_ksa_phone(self):
        redacted, stats = self.scrubber.scrub("اتصال: 055-123-4567")
        assert "[REDACTED_PHONE]" in redacted
        assert stats.phones_found > 0

    def test_uae_phone(self):
        redacted, stats = self.scrubber.scrub("الإمارات: +971-50-123-4567")
        assert "[REDACTED_PHONE]" in redacted

    def test_no_pii_untouched(self):
        original = "تقرير طبي لا يحوي أي معلومات شخصية إطلاقاً"
        redacted, stats = self.scrubber.scrub(original)
        assert redacted == original
        assert stats.total_redactions == 0


# ─── طبقة Regex: التواريخ والبريد والهويات ───────────────────────────────────

class TestRegexDatesEmailsIds:
    def setup_method(self):
        self.scrubber = HybridPIIScrubber(use_ner=False)

    def test_gregorian_date(self):
        redacted, stats = self.scrubber.scrub("تاريخ الميلاد: 15/03/1985")
        assert "[REDACTED_DATE]" in redacted
        assert stats.dates_found > 0

    def test_arabic_numeral_date(self):
        redacted, stats = self.scrubber.scrub("التاريخ: ٢٥/٠٦/٢٠٢٥")
        assert "[REDACTED_DATE]" in redacted

    def test_email(self):
        redacted, stats = self.scrubber.scrub("بريد التواصل: patient@example.com")
        assert "[REDACTED_EMAIL]" in redacted
        assert stats.emails_found > 0

    def test_national_id(self):
        redacted, stats = self.scrubber.scrub("الرقم الوطني: 1234567890")
        assert "[REDACTED_ID]" in redacted
        assert stats.ids_found > 0

    def test_medical_dosage_not_redacted(self):
        """جرعات الأدوية يجب ألا تُحجب كأرقام هوية (حماية إيجابية)."""
        text = "الوصفة: Amoxicillin 500mg ثلاث مرات يومياً"
        redacted, stats = self.scrubber.scrub(text)
        assert "500mg" in redacted
        assert "[REDACTED_ID]" not in redacted
        assert stats.ids_found == 0

    def test_multiple_pii_types(self):
        text = "المريض أحمد، الهاتف: +963-911-234-567، الميلاد: 15/03/1985، البريد: test@test.com"
        redacted, stats = self.scrubber.scrub(text)
        assert stats.total_redactions >= 3


# ─── الدفعات والسجلات ────────────────────────────────────────────────────────

class TestBatchAndRecords:
    def test_batch_processing(self):
        scrubber = HybridPIIScrubber(use_ner=False)
        results = scrubber.scrub_batch([
            "الهاتف: +963-911-234-567",
            "البريد: test@example.com",
            "لا معلومات هنا",
        ])
        assert len(results) == 3
        assert results[2][1].total_redactions == 0

    def test_scrub_records_preserves_non_text_fields(self):
        """حقول الميتاداتا تبقى حرفيًا — الحجب للنصوص فقط."""
        scrubber = HybridPIIScrubber(use_ner=False)
        records = [
            {"id": 1, "source": "تقرير-001", "text": "الهاتف: 055-123-4567"},
            {"id": 2, "source": "تقرير-002", "text": "بريد: a@b.com", "notes": None},
        ]
        out, stats = scrubber.scrub_records(records)
        assert out[0]["id"] == 1
        assert out[0]["source"] == "تقرير-001"
        assert out[0]["text"] == "الهاتف: [REDACTED_PHONE]"
        assert out[1]["text"] == "بريد: [REDACTED_EMAIL]"
        assert out[1]["notes"] is None
        assert stats.phones_found == 1
        assert stats.emails_found == 1

    def test_scrub_records_ocr_text_key(self):
        scrubber = HybridPIIScrubber(use_ner=False)
        records = [{"ocr_text": "الرقم الوطني: 1234567890"}]
        out, stats = scrubber.scrub_records(records)
        assert out[0]["ocr_text"] == "الرقم الوطني: [REDACTED_ID]"
        assert stats.ids_found == 1


# ─── فحص البقايا ────────────────────────────────────────────────────────────

class TestCheckResidue:
    def setup_method(self):
        self.scrubber = HybridPIIScrubber(use_ner=False)

    def test_residue_after_scrub_is_clean(self):
        text = "الهاتف: 0933-111-222، البريد: x@y.com، التاريخ: 12/05/2023"
        redacted, _ = self.scrubber.scrub(text)
        assert self.scrubber.check_residue(redacted) == {}

    def test_residue_detects_unscrubbed_text(self):
        residue = self.scrubber.check_residue("اتصل بالرقم 055-123-4567 فوراً")
        assert residue.get("phones_found", 0) > 0 or residue.get("[REDACTED_PHONE]", 0) > 0


# ─── طبقة NER: تعطيل رشيق بلا شبكة ──────────────────────────────────────────

class TestNERLayer:
    def test_ner_disabled_means_no_names_found(self):
        scrubber = HybridPIIScrubber(use_ner=False)
        redacted, stats = scrubber.scrub("المريض: أحمد الرشيد")
        assert stats.names_found == 0

    def test_ner_lazy_loading(self):
        """لا يُحمَّل أي نموذج عند البناء — التحميل عند أول استخدام فعلي."""
        scrubber = HybridPIIScrubber(use_ner=True)
        assert scrubber._ner_loaded is False
        assert scrubber._ner_pipeline is None

    def test_ner_graceful_degradation_without_transformers(self, monkeypatch):
        """غياب transformers → تعطيل رشيق، لا استثناء، مسح Regex مستمر."""
        monkeypatch.setitem(sys.modules, "transformers", None)
        scrubber = HybridPIIScrubber(use_ner=True)
        redacted, stats = scrubber.scrub("الهاتف: 055-123-4567")
        assert scrubber._ner_pipeline is None
        assert scrubber.use_ner is False
        assert "[REDACTED_PHONE]" in redacted  # الطبقة 1 عملت رغم فشل NER


# ─── الإحصاءات والدالة الراحة ────────────────────────────────────────────────

class TestStatsAndConvenience:
    def test_stats_dict(self):
        stats = RedactionStats(phones_found=2, dates_found=1, emails_found=1)
        assert stats.to_dict()["phones"] == 2
        assert stats.to_dict()["total"] == 4

    def test_stats_aggregation(self):
        total = RedactionStats()
        total.add(RedactionStats(phones_found=2))
        total.add(RedactionStats(phones_found=3, residue={"phones_found": 1}))
        assert total.phones_found == 5
        assert total.residue == {"phones_found": 1}

    def test_scrub_pii_convenience(self):
        redacted, stats = scrub_pii("الهاتف: +963-911-234-567", use_ner=False)
        assert "[REDACTED_PHONE]" in redacted
        assert stats.phones_found == 1


# ─── سياسة الوحدة ────────────────────────────────────────────────────────────

class TestPolicy:
    def test_policy_forbids_git_and_public_hf(self):
        """القاعدة المقصورة: بيانات المرضى أبدًا في git أو HF عام."""
        assert "git" in PRIVACY_POLICY["forbidden_locations"]
        assert "public-hf" in PRIVACY_POLICY["forbidden_locations"]
        assert "hf-private" in PRIVACY_POLICY["allowed_locations"]
        assert "hf-gated" in PRIVACY_POLICY["allowed_locations"]

    def test_policy_references_the_dataset(self):
        assert PRIVACY_POLICY["dataset"] == "DrAbdulmalek/arabic-medical-ocr-corrections"


def test_ner_model_resolution_is_local_only(monkeypatch):
    """حتى عند طلب NER، لا يُسمح بتنزيل نموذج من الشبكة."""
    import types

    captured = {}
    def fake_pipeline(task, **kwargs):
        captured["task"] = task
        captured.update(kwargs)
        return lambda text: []

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(pipeline=fake_pipeline),
    )
    scrubber = HybridPIIScrubber(use_ner=True)
    scrubber.scrub("نص اصطناعي لا يحتوي بيانات شخصية")
    assert captured["task"] == "ner"
    assert captured["model"] == scrubber.ner_model_name
    assert captured["model_kwargs"] == {"local_files_only": True}
