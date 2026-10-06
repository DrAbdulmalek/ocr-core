"""Graphics scene used by the medical region editor."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene

from .models import MedicalRegionData
from .region_item import MedicalRegionItem


class MedicalGraphicsScene(QGraphicsScene):
    region_selected = Signal(str)
    reocr_requested = Signal(str)
    region_deleted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.region_items: dict[str, MedicalRegionItem] = {}
        self.selectionChanged.connect(self._emit_selection)

    def set_document_image(self, pixmap: QPixmap) -> None:
        self.clear()
        self.region_items.clear()
        self.pixmap_item = self.addPixmap(pixmap)
        self.pixmap_item.setZValue(-100)
        self.setSceneRect(QRectF(pixmap.rect()))

    def add_region(self, region_data: MedicalRegionData) -> MedicalRegionItem:
        item = MedicalRegionItem(
            region_id=region_data.region_id,
            rect=QRectF(*region_data.bbox),
            region_type=region_data.region_type.value,
        )
        item.region_updated.connect(self._on_region_updated)
        item.reocr_requested.connect(self.reocr_requested.emit)
        item.region_deleted.connect(self._delete_region)
        self.addItem(item)
        self.region_items[region_data.region_id] = item
        return item

    def remove_region(self, region_id: str) -> None:
        item = self.region_items.pop(region_id, None)
        if item is not None:
            self.removeItem(item)
            item.deleteLater()

    def _delete_region(self, region_id: str) -> None:
        self.remove_region(region_id)
        self.region_deleted.emit(region_id)

    def _on_region_updated(self, region_id: str, rect: QRectF, region_type: str) -> None:
        self.update()

    def _emit_selection(self) -> None:
        selected = [item for item in self.selectedItems() if isinstance(item, MedicalRegionItem)]
        if selected:
            self.region_selected.emit(selected[0].region_id)
