from types import SimpleNamespace
import sys

from ocr_core.engines.indra import IndraEngine


def _page(text="مرحبا medical"):
    return SimpleNamespace(
        text=text,
        markdown=f"## {text}",
        rotation=0,
        skew=0.0,
        source="ocr",
        lines=[
            SimpleNamespace(text=text, conf=0.9, box=[1, 2, 30, 12]),
            SimpleNamespace(text="line two", conf=0.7, box=[1, 14, 30, 24]),
        ],
        blocks=[{"label": "paragraph", "text": text, "bbox": [1, 2, 30, 24]}],
    )


def test_indra_process_image_maps_text_confidence_and_layout(monkeypatch, tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"test")
    converter = SimpleNamespace(convert=lambda path: iter([(0, _page())]))
    fake_module = SimpleNamespace(load=lambda **kwargs: converter)
    monkeypatch.setitem(sys.modules, "indra_ocr", fake_module)

    result = IndraEngine(threads=2).process_image(image)

    assert result.ok
    assert result.engine == "indra"
    assert result.text == "مرحبا medical"
    assert result.confidence == 0.8
    assert result.pages == 1
    assert result.meta["runtime"] == "onnxruntime"
    assert result.meta["page_results"][0]["blocks"][0]["label"] == "paragraph"


def test_indra_process_pdf_respects_max_pages(monkeypatch, tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"test")
    converter = SimpleNamespace(
        convert=lambda path: iter([(0, _page("one")), (1, _page("two"))])
    )
    monkeypatch.setitem(sys.modules, "indra_ocr", SimpleNamespace(load=lambda **kwargs: converter))

    result = IndraEngine().process_pdf(pdf, max_pages=1)

    assert result.ok
    assert result.pages == 1
    assert result.text == "one"


def test_indra_missing_input_is_visible(tmp_path):
    result = IndraEngine().process_image(tmp_path / "missing.png")
    assert not result.ok
    assert result.error
    assert result.pages == 0


def test_indra_does_not_invent_confidence(monkeypatch, tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"test")
    page = SimpleNamespace(text="plain text", markdown=None, lines=[], blocks=[])
    converter = SimpleNamespace(convert=lambda path: iter([(0, page)]))
    monkeypatch.setitem(sys.modules, "indra_ocr", SimpleNamespace(load=lambda **kwargs: converter))

    result = IndraEngine().process_image(image)
    assert result.ok
    assert result.confidence == 0.0
