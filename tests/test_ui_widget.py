import importlib.util

import pytest

from ocr_core.ui.models import LayoutDocument, MedicalRegionData, RegionType

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None,
    reason="PySide6 optional dependency not installed",
)


def test_editor_binds_regions_to_document(qtbot):
    from PySide6.QtGui import QPixmap
    from ocr_core.ui.region_editor import RegionEditorWidget

    document = LayoutDocument(
        "DOC-UI",
        (800, 600),
        regions=[
            MedicalRegionData(
                region_id="r1",
                region_type=RegionType.MEDICAL_TABLE,
                bbox=(20, 30, 200, 100),
                text_verbatim="raw",
            )
        ],
    )
    editor = RegionEditorWidget()
    qtbot.addWidget(editor)
    editor.set_document(document, QPixmap(800, 600))

    assert len(editor.scene.region_items) == 1
    assert editor.scene.region_items["r1"].region_type is RegionType.MEDICAL_TABLE

    editor.scene.region_items["r1"].set_region_type(RegionType.CLINICAL_SECTION.value)
    assert document.regions[0].region_type is RegionType.CLINICAL_SECTION

    item = editor.scene.region_items["r1"]
    item.setPos(35, 45)
    assert document.regions[0].bbox[:2] == (55.0, 75.0)

    editor.scene.remove_region("r1")
    assert document.regions == []
