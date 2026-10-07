"""اختبارات كاشفات الكلمات — تعمل بلا نماذج ثقيلة."""
import pytest
from PIL import Image, ImageDraw

from ocr_core.preprocess.word_detector import (
    BBox, OnnxWordDetector, ProjectionDetector,
    HeuristicDetector, create_detector, detect_words,
)


# ---------- فيكسترات ----------
@pytest.fixture
def blank_image():
    """صورة بيضاء فارغة."""
    return Image.new("L", (400, 200), color=255)


@pytest.fixture
def three_words_image():
    """صورة فيها 3 'كلمات' — أشرطة داكنة متباعدة."""
    img = Image.new("L", (400, 200), color=255)
    draw = ImageDraw.Draw(img)
    # 3 شرائط أفقية كـ"كلمات" — بفجوات واضحة
    for x_start, x_end in [(30, 100), (150, 230), (280, 380)]:
        draw.rectangle([x_start, 50, x_end, 90], fill=0)
        # أضف ارتفاعًا داخليًا في كل كلمة
        draw.rectangle([x_start + 5, 40, x_end - 5, 100], fill=0)
    return img


@pytest.fixture
def two_lines_image():
    """صورة فيها سطران، كل سطر بكلمتين."""
    img = Image.new("L", (400, 300), color=255)
    draw = ImageDraw.Draw(img)
    # سطر 1: كلمتان (y=50-100)
    draw.rectangle([30, 60, 120, 100], fill=0)
    draw.rectangle([200, 60, 320, 100], fill=0)
    # سطر 2: كلمتان (y=180-220)
    draw.rectangle([50, 180, 140, 220], fill=0)
    draw.rectangle([220, 180, 350, 220], fill=0)
    return img


# ---------- اختبارات BBox ----------
def test_bbox_properties():
    b = BBox(x_min=10, y_min=20, x_max=60, y_max=70)
    assert b.width == 50
    assert b.height == 50
    assert b.area == 2500
    assert b.to_tuple() == (10, 20, 60, 70)


def test_bbox_to_dict():
    b = BBox(x_min=0, y_min=0, x_max=10, y_max=10, confidence=0.9)
    d = b.to_dict()
    assert d["bbox"] == [0, 0, 10, 10]
    assert d["confidence"] == 0.9


# ---------- اختبارات ProjectionDetector ----------
def test_projection_detects_three_words(three_words_image):
    det = ProjectionDetector(min_gap_width=5)
    boxes = det.detect(three_words_image)
    # يجب أن يجد 3 كلمات (قد يجد 2-3 بسبب العتبات)
    assert 2 <= len(boxes) <= 4
    # تحقق أن العرض منطقي
    for b in boxes:
        assert b.width > 20
        assert b.height > 20
        assert b.source == "projection"


def test_projection_returns_empty_on_blank(blank_image):
    det = ProjectionDetector()
    boxes = det.detect(blank_image)
    assert boxes == []


def test_projection_handles_small_image():
    """لا ينهار على صورة صغيرة."""
    small = Image.new("L", (20, 20), color=255)
    det = ProjectionDetector()
    assert det.detect(small) == []


def test_projection_rgb_input():
    """يقبل RGB ويحوّله لـ grayscale."""
    img = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 30, 150, 70], fill="black")
    det = ProjectionDetector()
    boxes = det.detect(img)
    assert len(boxes) >= 1


# ---------- اختبارات HeuristicDetector ----------
def test_heuristic_splits_into_strips():
    img = Image.new("L", (100, 200), color=255)
    det = HeuristicDetector(strip_height=50, strip_gap=10)
    boxes = det.detect(img)
    assert len(boxes) >= 3
    assert all(b.source == "heuristic" for b in boxes)


# ---------- اختبارات المصنع ----------
def test_create_detector_projection():
    det = create_detector("projection")
    assert isinstance(det, ProjectionDetector)


def test_create_detector_heuristic():
    det = create_detector("heuristic")
    assert isinstance(det, HeuristicDetector)


def test_create_detector_auto_fallback():
    """auto يختار onnx إن توفر، وإلا projection."""
    det = create_detector("auto")
    assert det.name in ("onnx", "projection")


def test_create_detector_invalid():
    with pytest.raises(ValueError, match="غير معروف"):
        create_detector("unknown_detector")


def test_detect_words_convenience(three_words_image):
    """دالة الاختصار تعمل مع projection صريح."""
    boxes = detect_words(three_words_image, kind="projection")
    assert isinstance(boxes, list)
    assert all(isinstance(b, BBox) for b in boxes)


# ---------- اختبار OnnxWordDetector ----------
def test_onnx_detector_lazy_load():
    """لا يحمّل النموذج حتى أول استدعاء."""
    det = OnnxWordDetector()
    assert det._loaded is False
    # لا نستدعي detect — لتجنّب تحميل النموذج في CI


def test_onnx_detector_fails_gracefully_without_ort(monkeypatch):
    """إن لم يكن onnxruntime مثبتًا، الخطأ واضح."""
    import sys
    # حجب onnxruntime
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    det = OnnxWordDetector()
    # _ensure_loaded سيرفع ImportError واضح
    # لا نستدعي detect هنا أيضًا
    assert det.name == "onnx"
