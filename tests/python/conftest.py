from io import BytesIO

import pytest
from PIL import Image, PngImagePlugin


@pytest.fixture
def png_bytes():
    """Build a 1x1 PNG carrying the given PNG text chunks."""

    def build(**fields: str) -> bytes:
        info = PngImagePlugin.PngInfo()
        for key, value in fields.items():
            info.add_text(key, value)
        buffer = BytesIO()
        Image.new("RGB", (1, 1)).save(buffer, "PNG", pnginfo=info)
        return buffer.getvalue()

    return build
