from io import BytesIO

from PIL import Image, PngImagePlugin

from slop_detector.metadata import inspect_image_metadata


def _png_with_text(**fields: str) -> bytes:
    info = PngImagePlugin.PngInfo()
    for key, value in fields.items():
        info.add_text(key, value)
    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, "PNG", pnginfo=info)
    return buffer.getvalue()


def test_metadata_reports_generator_software_but_not_absence():
    assert inspect_image_metadata(_png_with_text(Software="Stable Diffusion")) == [
        "Software: Stable Diffusion"
    ]
    assert inspect_image_metadata(_png_with_text()) == []


def test_metadata_ignores_unreadable_bytes():
    assert inspect_image_metadata(b"not an image") == []


def test_metadata_ignores_blank_generator_fields():
    assert inspect_image_metadata(_png_with_text(Software="   ")) == []


def test_metadata_reports_every_known_generator_field_sorted():
    data = _png_with_text(parameters="steps: 30", Generator="Midjourney")

    assert inspect_image_metadata(data) == [
        "Generator: Midjourney",
        "parameters: steps: 30",
    ]
