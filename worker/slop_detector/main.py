"""Worker entry point: one JSON report on stdout, everything else on stderr."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .contracts import RunReport, RunRequest
from .models import (
    TransformersLoader,
    run_image_detector,
    run_text_detectors,
    select_runtime_device,
)
from .webarchive import parse_webarchive


TOKEN_RELATIVE_PATH = Path(".secrets/huggingface.token")

ADVISORY_NOTES = (
    "Detector scores are advisory: they are not proof of authorship and are "
    "unsuitable as the sole basis for a high-stakes decision.",
    "Scores are reported per model on purpose; they are never merged into one "
    "AI percentage.",
    "EditLens reports an estimated AI-edit extent, not the probability that a "
    "page was written by a model.",
    "Each text score is the mean of that model's chunk scores; the chunk count "
    "beside it is that model's own, since each is chunked to its token window.",
)

IMAGE_NOTES = (
    "Organika/sdxl-detector recognizes SDXL-like generated imagery, so its "
    "confidence is not a universal AI-image provenance determination.",
    "Metadata findings are container heuristics, not detector predictions: a "
    "missing generator field is neutral, not evidence of human origin.",
)


def read_hf_token() -> str:
    """Return the Hugging Face token, read only from the gitignored secret file.

    The token value is never included in an error message.
    """
    for directory in [Path.cwd(), *Path.cwd().parents]:
        candidate = directory / TOKEN_RELATIVE_PATH
        if not candidate.is_file():
            continue
        try:
            token = candidate.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise RuntimeError(
                f"Hugging Face token unreadable at {TOKEN_RELATIVE_PATH}: "
                f"{type(error).__name__}"
            ) from None
        if token:
            return token
        break
    raise RuntimeError(
        f"Hugging Face token missing: place it in {TOKEN_RELATIVE_PATH}"
    )


def build_report(request: RunRequest) -> RunReport:
    """Parse the archive and run the requested detectors sequentially."""
    content = parse_webarchive(Path(request.archive_path))
    device, warnings = select_runtime_device()
    token = read_hf_token()

    text = run_text_detectors(
        content.text, TransformersLoader(token), device, request.verbose
    )
    if text and all(result.score is None for result in text):
        raise RuntimeError(
            "No text detector could run:\n"
            + "\n".join(f"{result.name}: {result.detail}" for result in text)
        )
    images = (
        run_image_detector(content.images, TransformersLoader(token), device)
        if request.include_images
        else []
    )

    notes = list(warnings)
    notes.extend(_unavailable_notes(text, images))
    if request.verbose:
        notes.extend(_verbose_notes(content, device, request))
    notes.extend(ADVISORY_NOTES)
    if request.include_images:
        notes.extend(IMAGE_NOTES)
    return RunReport(text=text, images=images, warnings=notes)


def _unavailable_notes(text: list, images: list) -> list[str]:
    """Say plainly that this is a partial report, and which detectors are missing."""
    missing = [result.name for result in text if result.score is None]
    if images and all(image.skipped_reason for image in images):
        missing.append("the image detector")
    if not missing:
        return []
    return [
        "This is a partial report: "
        + ", ".join(missing)
        + " could not run. The remaining results are unaffected."
    ]


def _verbose_notes(content: object, device: str, request: RunRequest) -> list[str]:
    """Report extraction counts without echoing any archive content."""
    embedded = len(content.images)
    notes = [
        f"Device: {device}.",
        f"Extracted {len(content.text)} characters of readable text.",
        f"Found {embedded} embedded raster image(s).",
    ]
    if embedded and not request.include_images:
        notes.append("Embedded images were not analyzed; pass --images to include them.")
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="slop_detector.main",
        description="Analyze a macOS .webarchive for likely AI-generated content.",
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    request = RunRequest(
        archive_path=str(args.archive),
        include_images=args.images,
        verbose=args.verbose,
    )
    try:
        report = build_report(request)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(report.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
