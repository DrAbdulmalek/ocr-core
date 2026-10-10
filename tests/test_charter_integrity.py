import os

def test_charter_v2_rules_exist():
    """يضمن وجود القواعد الحرجة في الميثاق"""
    charter_path = os.path.join(os.path.dirname(__file__), "..", "ocr_core", "charter_v2.yaml")
    with open(charter_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "ARABIC_MEDICAL_SANCTITY" in content, "قاعدة الحماية الطبية مفقودة من الميثاق"
    assert "NO_DESTRUCTIVE_NORMALIZATION" in content, "قاعدة منع التطبيع التدميري مفقودة"
    assert "ENGINE_VOTING" in content, "قاعدة التصويت بين المحركات مفقودة"

    assert "VISUAL_EVIDENCE_REQUIRED" in content
    assert "require_visual_evidence: true" in content
    assert 'default_without_evidence: "uncertain"' in content
    assert "AMBIGUOUS_MARKERS_REQUIRE_CONTEXT" in content
    assert "VERBATIM_AND_NORMALIZED_SEPARATE" in content
    assert "TRAINING_REQUIRES_EXPLICIT_CONFIRMATION" in content
