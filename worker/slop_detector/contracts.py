from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json


@dataclass(frozen=True)
class RunRequest:
    archive_path: str
    include_images: bool
    verbose: bool


@dataclass(frozen=True)
class DetectorResult:
    name: str
    #: None when the detector could not run; `detail` then carries the reason.
    score: float | None
    label: str
    detail: str
    #: Per-chunk scores behind `score`, so the spread stays inspectable and
    #: detectors can be compared chunk by chunk. Empty when nothing was scored.
    chunk_scores: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ImageResult:
    index: int
    score: float | None
    label: str | None
    metadata_flags: list[str]
    skipped_reason: str | None


@dataclass(frozen=True)
class RunReport:
    text: list[DetectorResult]
    images: list[ImageResult]
    warnings: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> RunReport:
        value = json.loads(raw)
        return cls(
            text=[DetectorResult(**item) for item in value["text"]],
            images=[ImageResult(**item) for item in value["images"]],
            warnings=value["warnings"],
        )
