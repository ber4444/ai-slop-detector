from __future__ import annotations

from io import BytesIO

from PIL import ExifTags, Image, UnidentifiedImageError


GENERATOR_FIELDS = {
    "software",
    "generator",
    "parameters",
    "prompt",
    "creatortool",
    "creator-tool",
}


def inspect_image_metadata(data: bytes) -> list[str]:
    """Return positive generator/software indicators found in image metadata.

    Only present, non-empty generator fields are reported: a missing field is
    neutral rather than evidence of human origin. These are heuristics about the
    container, not detector predictions about the pixels.
    """
    try:
        with Image.open(BytesIO(data)) as image:
            values = _collect_metadata(image)
    except (OSError, UnidentifiedImageError, ValueError):
        return []

    return sorted(
        f"{key}: {value.strip()}"
        for key, value in values.items()
        if _is_generator_field(key) and value.strip()
    )


def _collect_metadata(image: Image.Image) -> dict[str, str]:
    """Gather PNG text chunks, EXIF tags, and XMP entries as flat strings."""
    values = {str(key): str(value) for key, value in image.info.items()}

    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - broken EXIF must not fail the run
        exif = {}
    for tag, value in exif.items():
        values[str(ExifTags.TAGS.get(tag, tag))] = str(value)

    try:
        xmp = image.getxmp()
    except Exception:  # noqa: BLE001 - broken/absent XMP must not fail the run
        xmp = {}
    values.update(_flatten_xmp(xmp))

    return values


def _flatten_xmp(node: object) -> dict[str, str]:
    """Flatten the nested dictionary Pillow returns for XMP packets."""
    flat: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str):
                flat[str(key)] = value
            else:
                flat.update(_flatten_xmp(value))
    elif isinstance(node, list):
        for item in node:
            flat.update(_flatten_xmp(item))
    return flat


def _is_generator_field(key: str) -> bool:
    return key.lower().rsplit(":", 1)[-1] in GENERATOR_FIELDS
