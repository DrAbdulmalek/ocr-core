"""Main optional PySide6 medical layout editor widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsView, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .graphics_scene import MedicalGraphicsScene
from .models import LayoutDocument, MedicalRegionData


class RegionEditorWidget(QWidget):
    """UI shell; OCR execution remains owned by ocr-core callers."""

    save_requested = Signal()
    reocr_requested = Signal(str)
    region_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = MedicalGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._document: LayoutDocument | None = None

        toolbar = QHBoxLayout()
        save = QPushButton("حفظ المخطط")
        save.clicked.connect(self.save_requested.emit)
        toolbar.addWidget(save)
        toolbar.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.view)

        self.scene.reocr_requested.connect(self.reocr_requested.emit)
        self.scene.region_selected.connect(self.region_selected.emit)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_document(self, document: LayoutDocument, image: QPixmap) -> None:
        self._document = document
        self.scene.set_document_image(image)
        for region in document.regions:
            self.scene.add_region(region)

    def document(self) -> LayoutDocument | None:
        return self._document

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.StandardKey.Save):
            self.save_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_R and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            selected = [item for item in self.scene.selectedItems() if hasattr(item, "region_id")]
            if selected:
                self.reocr_requested.emit(selected[0].region_id)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):\n            items = sorted((item for item in self.scene.region_items.values()), key=lambda item: (item.scene_rect().top(), item.scene_rect().left()))\n            if items:\n                current = self.scene.selectedItems()[0] if self.scene.selectedItems() else None\n                step = -1 if event.key() == Qt.Key.Key_Backtab else 1\n                index = items.index(current) if current in items else (-1 if step > 0 else 0)\n                items[(index + step) % len(items)].setSelected(True)\n                event.accept()\n                return\n        if event.key() == Qt.Key.Key_Delete:
            selected = [item for item in self.scene.selectedItems() if hasattr(item, "region_id")]
            for item in selected:
                self.scene.remove_region(item.region_id)
            event.accept()
            return
        super().keyPressEvent(event)
