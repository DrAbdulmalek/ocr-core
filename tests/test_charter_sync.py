"""Drift guard for the bundled default rules.

Canonical source: marathon_ted_pipeline config/marathon_ocr_rules.yaml
(Visual Evidence Charter v2) + config/text_normalization_rules.yaml
(v1 18-rule pre-pass). The pre-pass remains byte-pinned. The charter is
checked semantically here because this PR deliberately tightens the v2
evidence gate; synchronize the canonical marathon copy in a follow-up PR.

History: a stale duplicate (src/ocr_core/rules/data/charter_v2.yaml holding
v1 content under a v2 name, orphaned by ef78be1) once shipped beside the
real defaults; the last test here keeps it from coming back.
"""
import hashlib
from pathlib import Path

import yaml

import ocr_core.rules.engine as engine_mod

_CONFIG = Path(engine_mod.__file__).resolve().parent.parent / "config"

PREPASS_SHA256 = "e5e8446824d25860ef3e003fe250ca7a46a335542f410608838ef2a1b48c8ebc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundled_charter_has_visual_evidence_gate_v2():
    charter = _CONFIG / "marathon_ocr_rules.yaml"
    assert charter.exists(), "bundled charter missing"
    cfg = yaml.safe_load(charter.read_text(encoding="utf-8"))
    assert cfg.get("version") == 2
    assert cfg["rules"][15]["params"]["require_visual_evidence"] is True
    for section in (
        "visual_markers", "color_semantics", "uncertainty_flags",
        "classification_labels", "training_data_rules", "final_verification",
    ):
        assert section in cfg, f"charter v2 section missing: {section}"


def test_bundled_prepass_is_pinned_v1_with_18_rules():
    prepass = _CONFIG / "text_normalization_rules.yaml"
    assert prepass.exists(), "bundled pre-pass missing"
    assert _sha256(prepass) == PREPASS_SHA256, (
        "bundled pre-pass drifted from the pinned canonical copy"
    )
    cfg = yaml.safe_load(prepass.read_text(encoding="utf-8"))
    ids = [r["id"] for r in cfg.get("rules", [])]
    assert ids == [f"R{i:02d}" for i in range(1, 19)]


def test_default_processor_resolves_inside_package_as_v2():
    proc = engine_mod.OCRProcessor()
    assert Path(proc.rules_file).resolve() == (_CONFIG / "marathon_ocr_rules.yaml").resolve()
    assert proc.version == 2
    assert len(proc.rules) == 18
    assert proc.normalization_file and Path(proc.normalization_file).exists()


def test_stale_orphan_duplicate_stays_removed():
    orphan = Path(engine_mod.__file__).resolve().parent / "data" / "charter_v2.yaml"
    assert not orphan.exists(), (
        "stale duplicate rules/data/charter_v2.yaml reintroduced — the real "
        "default lives in ocr_core/config/; keep a single source of truth"
    )
