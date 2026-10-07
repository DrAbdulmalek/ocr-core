"""تجميع الكلمات المكتشفة في سطور وترتيبها بترتيب القراءة."""
from __future__ import annotations

import logging
from typing import Optional

from .word_detector import BBox

logger = logging.getLogger(__name__)


class LineGrouper:
    """يجمع الكلمات في سطور ويرتّبها حسب الاتجاه."""

    def __init__(
        self,
        vertical_overlap_ratio: float = 0.5,
        line_gap_factor: float = 1.5,
    ):
        """
        Args:
            vertical_overlap_ratio: نسبة التداخل الرأسي لاعتبار كلمتين
                في نفس السطر.
            line_gap_factor: مضاعف متوسط ارتفاع السطر لاعتبار فراغ
                بين السطور.
        """
        self.vertical_overlap_ratio = vertical_overlap_ratio
        self.line_gap_factor = line_gap_factor

    def group(
        self,
        boxes: list[BBox],
        image_size: Optional[tuple[int, int]] = None,
        reading_direction: str = "rtl",
    ) -> list[list[BBox]]:
        """يرجع قائمة سطور، كل سطر قائمة BBox مرتّبة.

        Args:
            boxes: مربعات الكلمات من أي كاشف.
            image_size: (width, height) — للتحقق.
            reading_direction: "rtl" للعربية، "ltr" للإنجليزية.
        """
        if not boxes:
            return []

        # 1. رتّب أولًا حسب y_min للمساعدة في التجميع
        sorted_boxes = sorted(boxes, key=lambda b: (b.y_min, b.x_min))

        # 2. جمّع في سطور
        lines: list[list[BBox]] = []

        for box in sorted_boxes:
            placed = False
            for line in lines:
                if self._belongs_to_line(box, line):
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])

        # 3. أعد ترتيب كل سطر حسب الاتجاه
        for i, line in enumerate(lines):
            if reading_direction == "rtl":
                line.sort(key=lambda b: -b.x_min)  # من اليمين لليسار
            else:
                line.sort(key=lambda b: b.x_min)

            # ثبّت indices
            for j, b in enumerate(line):
                b.line_index = i
                b.word_index = j

        # 4. رتّب السطور من الأعلى للأسفل
        lines.sort(key=lambda line: min(b.y_min for b in line))
        for i, line in enumerate(lines):
            for j, b in enumerate(line):
                b.line_index = i
                b.word_index = j

        return lines

    def _belongs_to_line(self, box: BBox, line: list[BBox]) -> bool:
        """هل هذا المربع في نفس سطر مجموعة الـ line؟"""
        # احسب المدى الرأسي للسطر
        line_y_min = min(b.y_min for b in line)
        line_y_max = max(b.y_max for b in line)
        line_height = line_y_max - line_y_min

        if line_height == 0:
            return abs(box.y_min - line_y_min) < 10

        # احسب التداخل الرأسي
        overlap_top = max(box.y_min, line_y_min)
        overlap_bottom = min(box.y_max, line_y_max)
        overlap = max(0, overlap_bottom - overlap_top)

        # نسبة التداخل بالنسبة لارتفاع المربع الصغير
        min_height = min(box.height, line_height)
        if min_height == 0:
            return False

        return (overlap / min_height) >= self.vertical_overlap_ratio

    def to_text(self, lines: list[list[BBox]],
                words_text: Optional[dict[int, str]] = None) -> str:
        """يحوّل السطور إلى نص (بعد OCR).

        Args:
            lines: قائمة السطور من group().
            words_text: قاموس {id(bbox): text} أو استخدم فهرس تسلسلي.

        Returns:
            نص كامل، كل سطر على حدة.
        """
        out_lines = []
        for line in lines:
            line_texts = []
            for b in line:
                if words_text:
                    key = id(b)
                    line_texts.append(words_text.get(key, ""))
                else:
                    line_texts.append("")
            out_lines.append(" ".join(t for t in line_texts if t))
        return "\n".join(out_lines)
