"""اختبارات تجميع الكلمات في سطور."""
import pytest
from ocr_core.preprocess.word_detector import BBox
from ocr_core.preprocess.line_grouper import LineGrouper


def _box(x1, y1, x2, y2):
    return BBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2, confidence=0.9)


def test_groups_three_words_one_line():
    """3 كلمات في سطر واحد → سطر واحد."""
    boxes = [
        _box(300, 50, 380, 90),   # أيمن
        _box(150, 50, 250, 90),   # وسط
        _box(30, 50, 120, 90),    # أيسر
    ]
    grouper = LineGrouper()
    lines = grouper.group(boxes, reading_direction="rtl")

    assert len(lines) == 1
    assert len(lines[0]) == 3
    # RTL: أول كلمة = أيمن
    assert lines[0][0].x_min == 300
    assert lines[0][2].x_min == 30


def test_groups_two_lines_two_words_each():
    """سطران × كلمتان = 4 كلمات في سطرين."""
    boxes = [
        # سطر 1
        _box(300, 50, 380, 90),
        _box(30, 50, 120, 90),
        # سطر 2
        _box(250, 180, 350, 220),
        _box(40, 180, 140, 220),
    ]
    grouper = LineGrouper()
    lines = grouper.group(boxes, reading_direction="rtl")

    assert len(lines) == 2
    assert all(len(line) == 2 for line in lines)
    # السطر الأول أعلى
    assert min(b.y_min for b in lines[0]) < min(b.y_min for b in lines[1])


def test_assigns_line_and_word_indices():
    """كل BBox يجب أن يحصل على line_index و word_index."""
    boxes = [
        _box(300, 50, 380, 90),
        _box(30, 50, 120, 90),
        _box(250, 180, 350, 220),
    ]
    grouper = LineGrouper()
    lines = grouper.group(boxes, reading_direction="rtl")

    for line in lines:
        for b in line:
            assert b.line_index is not None
            assert b.word_index is not None


def test_ltr_direction():
    """LTR: أول كلمة = أيسر."""
    boxes = [
        _box(30, 50, 120, 90),
        _box(150, 50, 250, 90),
        _box(300, 50, 380, 90),
    ]
    grouper = LineGrouper()
    lines = grouper.group(boxes, reading_direction="ltr")

    assert lines[0][0].x_min == 30
    assert lines[0][2].x_min == 300


def test_empty_input():
    grouper = LineGrouper()
    assert grouper.group([]) == []


def test_to_text_with_word_map():
    boxes = [
        _box(300, 50, 380, 90),
        _box(30, 50, 120, 90),
    ]
    grouper = LineGrouper()
    lines = grouper.group(boxes, reading_direction="rtl")

    word_text = {id(boxes[0]): "مرحبا", id(boxes[1]): "بالعالم"}
    text = grouper.to_text(lines, word_text)
    assert "مرحبا" in text
    assert "بالعالم" in text
