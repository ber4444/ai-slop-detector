from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import plistlib

from bs4 import BeautifulSoup


RASTER_MIME_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


@dataclass(frozen=True)
class EmbeddedImage:
    index: int
    mime_type: str
    data: bytes


@dataclass(frozen=True)
class ArchiveContent:
    text: str
    images: list[EmbeddedImage]


def parse_webarchive(path: Path) -> ArchiveContent:
    try:
        archive = plistlib.loads(path.read_bytes())
        resource = archive["WebMainResource"]
        html = resource["WebResourceData"].decode("utf-8", errors="replace")
    except (
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        plistlib.InvalidFileException,
    ) as error:
        raise ValueError(f"Invalid webarchive: {error}") from error

    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select("script, style, noscript, nav, footer, aside, form"):
        node.decompose()
    root = soup.select_one("article, main, body") or soup
    text = "\n".join(line.strip() for line in root.stripped_strings)
    return ArchiveContent(
        text=text,
        images=_extract_raster_images(archive.get("WebSubresources", [])),
    )


def _extract_raster_images(resources: object) -> list[EmbeddedImage]:
    if not isinstance(resources, list):
        return []

    images: list[EmbeddedImage] = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        mime_type = resource.get("WebResourceMIMEType")
        data = resource.get("WebResourceData")
        if mime_type in RASTER_MIME_TYPES and isinstance(data, bytes):
            images.append(EmbeddedImage(len(images), mime_type, data))
    return images
