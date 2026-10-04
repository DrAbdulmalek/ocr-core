import os

def test_charter_v2_rules_exist():
    """يضمن وجود القواعد الحرجة في الميثاق"""
    charter_path = os.path.join(os.path.dirname(__file__), "..", "ocr_core", "charter_v2.yaml")
    with open(charter_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "ARABIC_MEDICAL_SANCTITY" in content, "قاعدة الحماية الطبية مفقودة من الميثاق"
    assert "NO_DESTRUCTIVE_NORMALIZATION" in content, "قاعدة منع التطبيع التدميري مفقودة"
    assert "ENGINE_VOTING" in content, "قاعدة التصويت بين المحركات مفقودة"
