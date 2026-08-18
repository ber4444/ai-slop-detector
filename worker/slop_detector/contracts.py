from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json


@dataclass(frozen=True)
class RunRequest:
    #: One or more Markdown input files.
    paths: list[str]
    verbose: bool


@dataclass(frozen=True)
class FileScore:
    """One detector's mean score for one file."""

    name: str
    #: None when the file held no readable prose.
    score: float | None
    chunks: int


@dataclass(frozen=True)
class DetectorResult:
    name: str
    #: Pooled mean over every chunk in every file; None when the detector could
    #: not run. In a multi-file survey this is the aggregate footnote, not the
    #: headline.
    score: float | None
    label: str
    detail: str
    #: Per-chunk scores behind `score`, kept so detectors can be compared chunk
    #: by chunk. Empty when nothing was scored.
    chunk_scores: list[float] = field(default_factory=list)
    #: Per-file mean scores, so a survey can count how many files each detector
    #: flags and chart the distribution.
    file_scores: list[FileScore] = field(default_factory=list)


@dataclass(frozen=True)
class RunReport:
    text: list[DetectorResult]
    warnings: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> RunReport:
        value = json.loads(raw)
        return cls(
            text=[
                DetectorResult(
                    name=item["name"],
                    score=item["score"],
                    label=item["label"],
                    detail=item["detail"],
                    chunk_scores=item.get("chunk_scores", []),
                    file_scores=[
                        FileScore(**file) for file in item.get("file_scores", [])
                    ],
                )
                for item in value["text"]
            ],
            warnings=value["warnings"],
        )
