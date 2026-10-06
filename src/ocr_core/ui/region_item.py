"""Interactive medical region item for the optional PySide6 UI."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QBrush, QPen, QPainter
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject, QMenu

from .models import RegionType


class MedicalRegionItem(QGraphicsObject):
    """Selectable/movable/resizable region with eight visual resize handles."""

    region_updated = Signal(str, QRectF, str)
    reocr_requested = Signal(str)
    region_deleted = Signal(str)

    COLORS = {
        RegionType.HEADER_CLINIC: "#3498db",
        RegionType.PATIENT_DEMOGRAPHICS: "#9b59b6",
        RegionType.CLINICAL_SECTION: "#2ecc71",
        RegionType.MEDICAL_TABLE: "#e67e22",
        RegionType.HANDWRITTEN_ANNOTATION: "#e74c3c",
        RegionType.FOOTER_SIGNATURE: "#16a085",
    }

    def __init__(self, region_id: str, rect: QRectF, region_type: str = RegionType.CLINICAL_SECTION.value):
        super().__init__()
        self.region_id = region_id
        self.region_type = RegionType(region_type)
        self._rect = QRectF(rect).normalized()
        self._drag_start: QPointF | None = None
        self._start_rect: QRectF | None = None
        self._handle = -1
        self.setFlags(
            self.GraphicsItemFlag.ItemIsSelectable
            | self.GraphicsItemFlag.ItemIsMovable
            | self.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-6, -6, 6, 6)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        color = QColor(self.COLORS[self.region_type])
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
        painter.drawRect(self._rect)
        if self.isSelected():
            painter.setBrush(QBrush(color))
            for point in self._handles():
                painter.drawRect(QRectF(point.x() - 3, point.y() - 3, 6, 6))

    def _handles(self) -> list[QPointF]:
        r = self._rect
        cx, cy = r.center().x(), r.center().y()
        return [
            QPointF(r.left(), r.top()), QPointF(cx, r.top()), QPointF(r.right(), r.top()),
            QPointF(r.right(), cy), QPointF(r.right(), r.bottom()), QPointF(cx, r.bottom()),
            QPointF(r.left(), r.bottom()), QPointF(r.left(), cy),
        ]

    def _hit_handle(self, pos: QPointF) -> int:
        return next((i for i, p in enumerate(self._handles()) if (p - pos).manhattanLength() <= 10), -1)

    def mousePressEvent(self, event) -> None:
        self._handle = self._hit_handle(event.pos()) if self.isSelected() else -1
        self._drag_start = event.pos()
        self._start_rect = QRectF(self._rect)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None or self._start_rect is None:
            return super().mouseMoveEvent(event)
        if self._handle < 0:
            return super().mouseMoveEvent(event)
        delta = event.pos() - self._drag_start
        r = QRectF(self._start_rect)
        if self._handle in (0, 1, 2): r.setTop(max(0.0, r.top() + delta.y()))
        if self._handle in (4, 5, 6): r.setBottom(max(r.top() + 1.0, r.bottom() + delta.y()))
        if self._handle in (0, 6, 7): r.setLeft(max(0.0, r.left() + delta.x()))
        if self._handle in (2, 3, 4): r.setRight(max(r.left() + 1.0, r.right() + delta.x()))
        self._rect = r.normalized()
        self.prepareGeometryChange()
        self.update()
        self.region_updated.emit(self.region_id, self.scene_rect(), self.region_type.value)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_start = None
        self._start_rect = None
        self._handle = -1
        self.region_updated.emit(self.region_id, self.scene_rect(), self.region_type.value)
        super().mouseReleaseEvent(event)

    def scene_rect(self) -> QRectF:
        return self.mapRectToScene(self._rect).normalized()

    def set_region_type(self, region_type: str) -> None:
        self.region_type = RegionType(region_type)
        self.update()
        self.region_updated.emit(self.region_id, self.scene_rect(), self.region_type.value)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu()
        type_menu = menu.addMenu("تغيير نوع المنطقة")
        for region_type in RegionType:
            action = type_menu.addAction(region_type.value)
            action.triggered.connect(lambda _checked=False, value=region_type.value: self.set_region_type(value))
        menu.addSeparator()
        reocr = menu.addAction("Re-OCR")
        reocr.triggered.connect(lambda: self.reocr_requested.emit(self.region_id))
        delete = menu.addAction("حذف المنطقة")
        delete.triggered.connect(lambda: self.region_deleted.emit(self.region_id))
        menu.exec(event.screenPos())
