"""Graphics scene used by the medical region editor."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene

from .models import LayoutDocument, MedicalRegionData
from .region_item import MedicalRegionItem


class MedicalGraphicsScene(QGraphicsScene):
    region_selected = Signal(str)
    reocr_requested = Signal(str)
    region_deleted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.region_items: dict[str, MedicalRegionItem] = {}
        self._document: LayoutDocument | None = None
        self.selectionChanged.connect(self._emit_selection)

    def set_document(self, document: LayoutDocument, pixmap: QPixmap) -> None:
        self._document = document
        self.clear()
        self.region_items.clear()
        self.pixmap_item = self.addPixmap(pixmap)
        self.pixmap_item.setZValue(-100)
        self.setSceneRect(QRectF(pixmap.rect()))
        for region in document.regions:
            self.add_region(region)

    def set_document_image(self, pixmap: QPixmap) -> None:
        if self._document is None:
            self._document = LayoutDocument(
                document_id="unsaved",
                image_dimensions=(pixmap.width(), pixmap.height()),
            )
        self.clear()
        self.region_items.clear()
        self.pixmap_item = self.addPixmap(pixmap)
        self.pixmap_item.setZValue(-100)
        self.setSceneRect(QRectF(pixmap.rect()))

    def add_region(self, region_data: MedicalRegionData) -> MedicalRegionItem:
        existing = self.region_items.get(region_data.region_id)
        if existing is not None:
            return existing
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
        if self._document is not None and not any(
            region.region_id == region_data.region_id for region in self._document.regions
        ):
            self._document.regions.append(region_data)
        return item

    def remove_region(self, region_id: str) -> None:
        item = self.region_items.pop(region_id, None)
        if item is not None:
            self.removeItem(item)
            item.deleteLater()
        if self._document is not None:
            self._document.regions = [
                region for region in self._document.regions if region.region_id != region_id
            ]
        self.region_deleted.emit(region_id)

    def _delete_region(self, region_id: str) -> None:
        self.remove_region(region_id)

    def _on_region_updated(self, region_id: str, rect: QRectF, region_type: str) -> None:
        if self._document is not None:
            for region in self._document.regions:
                if region.region_id == region_id:
                    region.bbox = (
                        rect.x(),
                        rect.y(),
                        rect.width(),
                        rect.height(),
                    )
                    region.region_type = region.region_type.__class__(region_type)
                    region.is_manually_edited = True
                    break
        self.update()

    def _emit_selection(self) -> None:
        selected = [
            item for item in self.selectedItems()
            if isinstance(item, MedicalRegionItem)
        ]
        if selected:
            self.region_selected.emit(selected[0].region_id)
