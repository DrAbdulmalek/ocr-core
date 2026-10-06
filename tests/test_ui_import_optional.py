import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("PySide6") is None, reason="PySide6 optional dependency not installed")
def test_region_editor_modules_import():
    from ocr_core.ui.graphics_scene import MedicalGraphicsScene
    from ocr_core.ui.region_editor import RegionEditorWidget
    from ocr_core.ui.region_item import MedicalRegionItem

    assert MedicalGraphicsScene is not None
    assert RegionEditorWidget is not None
    assert MedicalRegionItem is not None
