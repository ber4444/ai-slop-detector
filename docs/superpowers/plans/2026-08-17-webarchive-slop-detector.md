# Webarchive Slop Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a Kotlin entry-point CLI that runs local Hugging Face text and optional image detectors over a `.webarchive` file and prints an advisory terminal report.

**Architecture:** `slop-detector.main.kts` validates CLI input and invokes a project-local Python virtual environment. The Python worker parses the webarchive, produces a JSON report on stdout, and owns all Hugging Face/Transformers execution with sequential MPS model loading. The Kotlin script renders that report without printing archive contents or credentials.

**Tech Stack:** Kotlin scripting, Python 3.11+, `pytest`, `beautifulsoup4`, `Pillow`, `torch`, `transformers`, `peft`, Hugging Face Hub.

**Spec:** `docs/superpowers/specs/2026-08-17-webarchive-slop-detector-design.md`

## Global Constraints

- Target an Apple M4 Mac mini with 24 GB RAM; use MPS when available and warn before CPU fallback.
- Run `pangram/editlens_Llama-3.2-3B`, `ogmatrixllm/glyph-v1.1`, and `ShantanuT01/vanguard-ai-text-detector` sequentially.
- `--images` alone enables `Organika/sdxl-detector` plus image metadata checks.
- All archive inference is local after model download; do not call remote inference APIs.
- Read the Hugging Face token only from `.secrets/huggingface.token`; ignore `.secrets/`, `.venv/`, and `.model-cache/` in Git.
- Keep image predictions, text predictions, and metadata heuristics separate; never produce one combined AI percentage.
- Treat all output as advisory; print the detector limitations and never use a detector result as a provenance assertion.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `.gitignore` | Exclude token, virtual environment, model cache, Python cache, and test artifacts. |
| `slop-detector.main.kts` | Kotlin command parsing, worker bootstrap/invocation, JSON decoding, human-readable rendering, exit-code propagation. |
| `worker/requirements.txt` | Reproducible Python dependencies for the local worker. |
| `worker/slop_detector/contracts.py` | Typed request/report structures and JSON serialization. |
| `worker/slop_detector/webarchive.py` | macOS webarchive plist parsing, readable text selection, and embedded-image extraction. |
| `worker/slop_detector/metadata.py` | Non-model image metadata heuristics. |
| `worker/slop_detector/models.py` | Device selection, chunking/aggregation, and sequential model inference adapters. |
| `worker/slop_detector/main.py` | Worker CLI orchestration and stable JSON/error contract. |
| `tests/python/` | Offline worker unit and contract tests using fixtures and fake model loaders. |
| `tests/kotlin/test_cli.py` | Black-box tests that run the Kotlin script against a fake Python executable/worker. |
| `tests/fixtures/` | Small XML plist `.webarchive` fixtures and encoded image bytes. |
| `README.md` | Install, gated-model authorization, token placement, normal use, and opt-in smoke-test instructions. |

### Task 1: Scaffold the local-first project contract

**Files:**
- Create: `.gitignore`
- Create: `worker/requirements.txt`
- Create: `worker/slop_detector/__init__.py`
- Create: `worker/slop_detector/contracts.py`
- Create: `tests/python/test_contracts.py`
- Create: `pyproject.toml`

**Interfaces:**
- Produces: `RunRequest(archive_path: str, include_images: bool, verbose: bool)`, `DetectorResult(name: str, score: float, label: str, detail: str)`, `ImageResult(index: int, score: float | None, label: str | None, metadata_flags: list[str], skipped_reason: str | None)`, and `RunReport(text: list[DetectorResult], images: list[ImageResult], warnings: list[str])`.
- Produces: `RunReport.to_json() -> str` and `RunReport.from_json(raw: str) -> RunReport` as the Kotlin/Python boundary.

- [x] **Step 1: Write the failing contract round-trip test**

```python
from slop_detector.contracts import DetectorResult, ImageResult, RunReport


def test_report_json_round_trip_preserves_separate_text_image_and_metadata_results():
    report = RunReport(
        text=[DetectorResult("glyph", 0.75, "likely AI-generated", "2 chunks")],
        images=[ImageResult(0, 0.9, "artificial", ["Software: Stable Diffusion"], None)],
        warnings=["Detector scores are advisory."],
    )

    parsed = RunReport.from_json(report.to_json())

    assert parsed == report
```

- [x] **Step 2: Run the test to verify it fails because the package is absent**

Run: `python3 -m pytest tests/python/test_contracts.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'slop_detector'`.

- [x] **Step 3: Add the minimal typed JSON contract and dependency files**

```python
# worker/slop_detector/contracts.py
from __future__ import annotations
from dataclasses import asdict, dataclass
import json

@dataclass(frozen=True)
class DetectorResult:
    name: str; score: float; label: str; detail: str

@dataclass(frozen=True)
class ImageResult:
    index: int; score: float | None; label: str | None
    metadata_flags: list[str]; skipped_reason: str | None

@dataclass(frozen=True)
class RunReport:
    text: list[DetectorResult]; images: list[ImageResult]; warnings: list[str]
    def to_json(self) -> str: return json.dumps(asdict(self), sort_keys=True)
    @classmethod
    def from_json(cls, raw: str) -> "RunReport":
        value = json.loads(raw)
        return cls([DetectorResult(**x) for x in value["text"]],
                   [ImageResult(**x) for x in value["images"]], value["warnings"])
```

```text
# .gitignore
.secrets/
.venv/
.model-cache/
__pycache__/
.pytest_cache/

# worker/requirements.txt
beautifulsoup4==4.13.4
Pillow==11.3.0
torch==2.7.1
transformers==4.53.0
peft==0.16.0
huggingface_hub==0.33.4

# pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["worker"]
testpaths = ["tests/python", "tests/kotlin"]
```

- [x] **Step 4: Run the contract test to verify it passes**

Run: `python3 -m pytest tests/python/test_contracts.py -q`

Expected: PASS with `1 passed`.

- [x] **Step 5: Commit the scaffold**

```bash
git add .gitignore pyproject.toml worker tests/python/test_contracts.py
git commit -m "chore: scaffold local detector worker"
```

### Task 2: Parse webarchive resources and readable text

**Files:**
- Create: `worker/slop_detector/webarchive.py`
- Create: `tests/fixtures/article-with-image.webarchive`
- Create: `tests/fixtures/malformed.webarchive`
- Create: `tests/python/test_webarchive.py`

**Interfaces:**
- Consumes: `Path` to a macOS XML or binary property-list webarchive.
- Produces: `ArchiveContent(text: str, images: list[EmbeddedImage])` where `EmbeddedImage(index: int, mime_type: str, data: bytes)`.
- Produces: `parse_webarchive(path: Path) -> ArchiveContent` and raises `ValueError` with the prefix `Invalid webarchive:` for malformed archives.

- [x] **Step 1: Write failing extraction tests using a minimal plist fixture**

```python
from pathlib import Path
import pytest
from slop_detector.webarchive import parse_webarchive

FIXTURES = Path(__file__).parents[1] / "fixtures"

def test_parse_webarchive_returns_readable_body_text_and_embedded_raster_images():
    content = parse_webarchive(FIXTURES / "article-with-image.webarchive")
    assert content.text == "Headline\nFirst paragraph.\nSecond paragraph."
    assert [(image.index, image.mime_type) for image in content.images] == [(0, "image/png")]

def test_parse_webarchive_rejects_non_plist_input():
    with pytest.raises(ValueError, match=r"^Invalid webarchive:"):
        parse_webarchive(FIXTURES / "malformed.webarchive")
```

- [x] **Step 2: Run the webarchive tests to verify they fail because the parser is absent**

Run: `python3 -m pytest tests/python/test_webarchive.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'slop_detector.webarchive'`.

- [x] **Step 3: Implement the plist/resource parser and content selection**

```python
@dataclass(frozen=True)
class EmbeddedImage:
    index: int; mime_type: str; data: bytes

@dataclass(frozen=True)
class ArchiveContent:
    text: str; images: list[EmbeddedImage]

def parse_webarchive(path: Path) -> ArchiveContent:
    try:
        archive = plistlib.loads(path.read_bytes())
        resource = archive["WebMainResource"]
        html = resource["WebResourceData"].decode("utf-8", errors="replace")
    except (KeyError, OSError, plistlib.InvalidFileException, TypeError) as error:
        raise ValueError(f"Invalid webarchive: {error}") from error
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select("script, style, noscript, nav, footer, aside, form"):
        node.decompose()
    root = soup.select_one("article, main, body") or soup
    text = "\n".join(line.strip() for line in root.stripped_strings)
    images = extract_raster_images(archive.get("WebSubresources", []))
    return ArchiveContent(text=text, images=images)
```

Create `article-with-image.webarchive` as an XML plist containing `WebMainResource/WebResourceData` for the exact HTML in the assertion and one `image/png` subresource whose base64 data is a 1×1 PNG. Create `malformed.webarchive` with the literal bytes `not a plist`.

- [x] **Step 4: Run the parser tests to verify they pass**

Run: `python3 -m pytest tests/python/test_webarchive.py -q`

Expected: PASS with `2 passed`.

- [x] **Step 5: Commit webarchive parsing**

```bash
git add worker/slop_detector/webarchive.py tests/fixtures tests/python/test_webarchive.py
git commit -m "feat: parse webarchive text and images"
```

### Task 3: Implement model-independent text scoring and image metadata heuristics

**Files:**
- Create: `worker/slop_detector/metadata.py`
- Create: `worker/slop_detector/models.py`
- Create: `tests/python/test_models.py`
- Create: `tests/python/test_metadata.py`

**Interfaces:**
- Produces: `select_device(torch_module) -> str`, `chunk_text(text: str, max_characters: int = 6000) -> list[str]`, `mean_score(scores: list[float]) -> float`.
- Produces: `editlens_score(logits: list[float]) -> float`, using `sum(index * softmax(logits)[index]) / (len(logits) - 1)`.
- Produces: `inspect_image_metadata(data: bytes) -> list[str]`, which returns only positive generator/software indicators.

- [x] **Step 1: Write failing pure-function tests for chunk boundaries, EditLens scoring, and metadata**

```python
from io import BytesIO
from PIL import Image, PngImagePlugin
from slop_detector.metadata import inspect_image_metadata
from slop_detector.models import chunk_text, editlens_score, select_device

def test_chunk_text_keeps_sentences_and_honors_the_limit():
    chunks = chunk_text("One. Two. Three.", max_characters=10)
    assert chunks == ["One. Two.", "Three."]

def test_editlens_score_is_normalized_expected_bucket_value():
    assert editlens_score([0.0, 0.0, 0.0]) == 0.5

def test_select_device_prefers_mps_then_cpu():
    mps = type("M", (), {"is_available": staticmethod(lambda: True)})()
    torch = type("Torch", (), {"backends": type("B", (), {"mps": mps})()})()
    assert select_device(torch) == "mps"

def test_metadata_reports_generator_software_but_not_absence():
    image = Image.new("RGB", (1, 1))
    png = PngImagePlugin.PngInfo(); png.add_text("Software", "Stable Diffusion")
    data = BytesIO(); image.save(data, "PNG", pnginfo=png)
    assert inspect_image_metadata(data.getvalue()) == ["Software: Stable Diffusion"]
    assert inspect_image_metadata(b"not an image") == []
```

- [x] **Step 2: Run pure-function tests to verify they fail because the modules are absent**

Run: `python3 -m pytest tests/python/test_models.py tests/python/test_metadata.py -q`

Expected: FAIL with `ModuleNotFoundError` for `slop_detector.models` and `slop_detector.metadata`.

- [x] **Step 3: Implement deterministic helpers without loading model weights**

```python
def select_device(torch_module: object) -> str:
    return "mps" if torch_module.backends.mps.is_available() else "cpu"

def editlens_score(logits: list[float]) -> float:
    probabilities = softmax(logits)
    return sum(index * probability for index, probability in enumerate(probabilities)) / (len(probabilities) - 1)

def inspect_image_metadata(data: bytes) -> list[str]:
    try:
        with Image.open(BytesIO(data)) as image:
            values = {**image.info, **{str(k): str(v) for k, v in image.getexif().items()}}
    except (OSError, UnidentifiedImageError):
        return []
    return [f"{key}: {value}" for key, value in values.items()
            if key.lower() in {"software", "generator", "parameters", "prompt"}
            and str(value).strip()]
```

Ensure `chunk_text` first splits on sentence endings, only splits a single overlong sentence as a last resort, drops empty chunks, and never returns a chunk longer than the limit. Ensure the score helper raises `ValueError` for fewer than two logits.

- [x] **Step 4: Run the pure-function tests to verify they pass**

Run: `python3 -m pytest tests/python/test_models.py tests/python/test_metadata.py -q`

Expected: PASS with `4 passed`.

- [x] **Step 5: Commit scoring and metadata helpers**

```bash
git add worker/slop_detector/models.py worker/slop_detector/metadata.py tests/python/test_models.py tests/python/test_metadata.py
git commit -m "feat: add scoring and image metadata helpers"
```

### Task 4: Add sequential local model adapters

**Files:**
- Modify: `worker/slop_detector/models.py`
- Modify: `tests/python/test_models.py`

**Interfaces:**
- Consumes: `ModelLoader` protocol with `load_text(model_id: str, device: str)` and `load_editlens(base_id: str, adapter_id: str, device: str)`.
- Produces: `run_text_detectors(text: str, loader: ModelLoader, device: str) -> list[DetectorResult]` in the stable order Glyph, Vanguard, EditLens.
- Produces: `run_image_detector(images: list[EmbeddedImage], loader: ModelLoader, device: str) -> list[ImageResult]`.

- [x] **Step 1: Write failing adapter tests with a fake loader**

```python
from slop_detector.models import run_text_detectors

class FakeLoader:
    def __init__(self): self.loaded = []
    def load_text(self, model_id, device):
        self.loaded.append(model_id)
        return lambda chunk: {"ai": 0.8}
    def load_editlens(self, base_id, adapter_id, device):
        self.loaded.append(adapter_id)
        return lambda chunk: [0.0, 0.0, 0.0]

def test_text_detectors_run_sequentially_and_keep_their_distinct_meaning():
    loader = FakeLoader()
    results = run_text_detectors("A short sentence.", loader, "mps")
    assert [result.name for result in results] == ["Glyph", "Vanguard", "EditLens"]
    assert [result.score for result in results] == [0.8, 0.8, 0.5]
    assert loader.loaded == [
        "ogmatrixllm/glyph-v1.1",
        "ShantanuT01/vanguard-ai-text-detector",
        "pangram/editlens_Llama-3.2-3B",
    ]
    assert results[2].label == "estimated AI-edit extent"
```

- [x] **Step 2: Run the adapter test to verify it fails because the runner is absent**

Run: `python3 -m pytest tests/python/test_models.py::test_text_detectors_run_sequentially_and_keep_their_distinct_meaning -q`

Expected: FAIL with `ImportError` for `run_text_detectors`.

- [x] **Step 3: Implement the fakeable adapters and production Hugging Face loader**

```python
TEXT_MODELS = (
    ("Glyph", "ogmatrixllm/glyph-v1.1", "likely AI-generated"),
    ("Vanguard", "ShantanuT01/vanguard-ai-text-detector", "likely AI-generated"),
)
EDITLENS_BASE = "meta-llama/Llama-3.2-3B"
EDITLENS_ADAPTER = "pangram/editlens_Llama-3.2-3B"

def run_text_detectors(text: str, loader: ModelLoader, device: str) -> list[DetectorResult]:
    chunks = chunk_text(text)
    results = []
    for name, model_id, label in TEXT_MODELS:
        predictor = loader.load_text(model_id, device)
        results.append(DetectorResult(name, mean_score([predictor(c)["ai"] for c in chunks]), label, f"{len(chunks)} chunks"))
        del predictor
        gc.collect()
    predictor = loader.load_editlens(EDITLENS_BASE, EDITLENS_ADAPTER, device)
    results.append(DetectorResult("EditLens", mean_score([editlens_score(predictor(c)) for c in chunks]), "estimated AI-edit extent", f"{len(chunks)} chunks"))
    return results
```

`TransformersLoader` must pass the token from `read_hf_token()` to every `from_pretrained` call, set models to evaluation mode, use `torch.inference_mode()`, and clear MPS cache after each adapter. It must derive text-classifier AI probability from each model’s `id2label` instead of assuming a fixed label index. For the Organika image model, read the `artificial` softmax probability and retain the metadata output from Task 3.

- [x] **Step 4: Run the adapter test to verify it passes**

Run: `python3 -m pytest tests/python/test_models.py::test_text_detectors_run_sequentially_and_keep_their_distinct_meaning -q`

Expected: PASS with `1 passed`.

- [x] **Step 5: Commit model adapters**

```bash
git add worker/slop_detector/models.py tests/python/test_models.py
git commit -m "feat: run local detector models sequentially"
```

### Task 5: Build the Python worker CLI and failure contract

**Files:**
- Create: `worker/slop_detector/main.py`
- Create: `tests/python/test_worker_cli.py`
- Modify: `worker/slop_detector/contracts.py`

**Interfaces:**
- Consumes: `python -m slop_detector.main --archive <path> [--images] [--verbose]`.
- Produces: exactly one `RunReport.to_json()` document on stdout and exit code `0` on success.
- Produces: errors written only to stderr, with exit code `2` for invalid input and `1` for setup/model failures.

- [x] **Step 1: Write failing worker CLI tests for normal and optional-image behavior**

```python
import json
from pathlib import Path
from slop_detector.main import main

def test_worker_emits_machine_readable_report_without_images(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("slop_detector.main.parse_webarchive", lambda _: type("C", (), {"text": "Text.", "images": [object()]})())
    monkeypatch.setattr("slop_detector.main.run_text_detectors", lambda *args: [])
    assert main(["--archive", str(tmp_path / "page.webarchive")]) == 0
    assert json.loads(capsys.readouterr().out)["images"] == []

def test_worker_reports_individual_bad_images_without_discarding_text(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("slop_detector.main.parse_webarchive", lambda _: type("C", (), {"text": "Text.", "images": [object()]})())
    monkeypatch.setattr("slop_detector.main.run_text_detectors", lambda *args: [])
    monkeypatch.setattr("slop_detector.main.run_image_detector", lambda *args: ["skipped"])
    assert main(["--archive", str(tmp_path / "page.webarchive"), "--images"]) == 0
    assert json.loads(capsys.readouterr().out)["images"] == ["skipped"]
```

- [x] **Step 2: Run worker CLI tests to verify they fail because the entry point is absent**

Run: `python3 -m pytest tests/python/test_worker_cli.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'slop_detector.main'`.

- [x] **Step 3: Implement argument parsing and report-only stdout**

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        content = parse_webarchive(args.archive)
        device, warnings = select_runtime_device()
        text = run_text_detectors(content.text, TransformersLoader(read_hf_token()), device)
        images = run_image_detector(content.images, TransformersLoader(read_hf_token()), device) if args.images else []
        print(RunReport(text, images, warnings).to_json())
        return 0
    except ValueError as error:
        print(str(error), file=sys.stderr); return 2
    except RuntimeError as error:
        print(str(error), file=sys.stderr); return 1
```

`read_hf_token()` must read and trim only `.secrets/huggingface.token`, reject an absent/empty file with `Hugging Face token missing: place it in .secrets/huggingface.token`, and never include token bytes in errors. Wrap gated-repository download failures with a message that names both model pages whose access terms must be accepted.

- [x] **Step 4: Run worker CLI tests to verify they pass**

Run: `python3 -m pytest tests/python/test_worker_cli.py -q`

Expected: PASS with `2 passed`.

- [x] **Step 5: Commit the worker CLI**

```bash
git add worker/slop_detector/main.py worker/slop_detector/contracts.py tests/python/test_worker_cli.py
git commit -m "feat: add local inference worker CLI"
```

### Task 6: Create and test the Kotlin user-facing command

**Files:**
- Create: `slop-detector.main.kts`
- Create: `tests/kotlin/test_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `kotlin slop-detector.main.kts <archive.webarchive> [--images] [--verbose]`.
- Produces: a readable terminal report with `Text detectors`, optional `Image detectors`, `Metadata heuristics`, and advisory text; returns worker exit status on failure.

- [x] **Step 1: Write a failing black-box test using a fake Python executable**

```python
import os, subprocess
from pathlib import Path

def test_kotlin_cli_renders_worker_report_and_forwards_images_flag(tmp_path):
    python = tmp_path / "python3"
    received = tmp_path / "received-args"
    python.write_text(f"#!/bin/sh\nprintf '%s' \\\"$*\\\" > {received}\nprintf '{{\\\"text\\\":[{{\\\"name\\\":\\\"Glyph\\\",\\\"score\\\":0.75,\\\"label\\\":\\\"likely AI-generated\\\",\\\"detail\\\":\\\"1 chunks\\\"}}],\\\"images\\\":[],\\\"warnings\\\":[\\\"Detector scores are advisory.\\\"]}}'\n")
    python.chmod(0o755)
    result = subprocess.run(
        ["kotlin", "slop-detector.main.kts", "page.webarchive", "--images"],
        text=True, capture_output=True,
        env={**os.environ, "SLOP_DETECTOR_PYTHON": str(python), "SLOP_DETECTOR_SKIP_BOOTSTRAP": "1"},
    )
    assert result.returncode == 0
    assert "Glyph: 75.0% — likely AI-generated" in result.stdout
    assert "Detector scores are advisory." in result.stdout
    assert "--images" in received.read_text()
```

- [x] **Step 2: Run the Kotlin CLI test to verify it fails because the script is absent**

Run: `python3 -m pytest tests/kotlin/test_cli.py -q`

Expected: FAIL with a nonzero process result and `script file not found` in stderr.

- [x] **Step 3: Implement the Kotlin script with explicit process handling**

```kotlin
#!/usr/bin/env kotlin
@file:DependsOn("org.jetbrains.kotlinx:kotlinx-serialization-json:1.8.1")

import java.io.File
import kotlinx.serialization.json.Json

val options = args.toList()
if (options == listOf("--help") || options == listOf("-h")) {
    println("Usage: kotlin slop-detector.main.kts <archive.webarchive> [--images] [--verbose]")
    kotlin.system.exitProcess(0)
}
val archive = options.firstOrNull { !it.startsWith("--") }
    ?: error("Usage: kotlin slop-detector.main.kts <archive.webarchive> [--images] [--verbose]")
require(archive.endsWith(".webarchive")) { "Expected a .webarchive file: $archive" }
val python = System.getenv("SLOP_DETECTOR_PYTHON") ?: ".venv/bin/python3"
val command = listOf(python, "-m", "slop_detector.main", "--archive", archive) + options.filter { it.startsWith("--") }
val process = ProcessBuilder(command).directory(File("worker")).redirectError(ProcessBuilder.Redirect.INHERIT).start()
val payload = process.inputStream.bufferedReader().readText()
val code = process.waitFor()
if (code != 0) kotlin.system.exitProcess(code)
renderReport(payload)
```

Before starting the worker, `bootstrapPython()` must create `.venv` with `python3 -m venv .venv`, install `worker/requirements.txt`, set `HF_HOME` to `<repo>/.model-cache`, and skip those commands only when `SLOP_DETECTOR_SKIP_BOOTSTRAP=1`. Implement `renderReport` with Kotlin’s `kotlinx.serialization-json` dependency declared at the top of the script, render scores with one decimal percent, print only the image/metadata sections when results exist, and append every warning under `Notes:`.

- [x] **Step 4: Run the Kotlin CLI test to verify it passes**

Run: `python3 -m pytest tests/kotlin/test_cli.py -q`

Expected: PASS with `1 passed`.

- [x] **Step 5: Commit the Kotlin CLI**

```bash
git add slop-detector.main.kts tests/kotlin/test_cli.py pyproject.toml
git commit -m "feat: add Kotlin webarchive detector CLI"
```

### Task 7: Document safe setup and complete verification

**Files:**
- Create: `README.md`
- Modify: `tests/python/test_worker_cli.py`

**Interfaces:**
- Documents exact token location, required Hugging Face terms acceptance, Kotlin/Python prerequisites, `--images`, model-cache behavior, and the intentional limitations of all detector results.

- [x] **Step 1: Write a failing test that ensures the token error gives the safe placement**

```python
from slop_detector.main import read_hf_token
import pytest

def test_missing_token_names_the_gitignored_token_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match=r"\.secrets/huggingface\.token"):
        read_hf_token()
```

- [x] **Step 2: Run the test to verify it fails if the safe error is missing**

Run: `python3 -m pytest tests/python/test_worker_cli.py::test_missing_token_names_the_gitignored_token_file -q`

Expected: FAIL until the exact safe path is part of `read_hf_token()`.

- [x] **Step 3: Complete the safe token failure and write README instructions**

```markdown
## Hugging Face access

Put a read token in `.secrets/huggingface.token` (one token only, no `HF_TOKEN=` prefix). This directory is excluded from Git. Before the first run, log into that Hugging Face account and accept access conditions for `meta-llama/Llama-3.2-3B` and `pangram/editlens_Llama-3.2-3B`.

## Run

kotlin slop-detector.main.kts saved-page.webarchive
kotlin slop-detector.main.kts saved-page.webarchive --images --verbose
```

Document model licenses: EditLens is CC BY-NC-SA 4.0 and Organika’s SDXL detector is CC BY-NC 3.0, so this repository’s intended use remains personal/non-commercial. State the explicit first-run download behavior and the opt-in smoke command `kotlin slop-detector.main.kts tests/fixtures/article-with-image.webarchive --images`.

- [x] **Step 4: Run the complete offline test suite**

Run: `python3 -m pytest -q`

Expected: PASS with no skipped or failing tests.

- [x] **Step 5: Compile-check the Kotlin script without bootstrapping downloads**

Run: `SLOP_DETECTOR_SKIP_BOOTSTRAP=1 SLOP_DETECTOR_PYTHON=/usr/bin/true kotlin slop-detector.main.kts --help`

Expected: Kotlin script parses and prints usage without creating a virtual environment or downloading weights.

- [x] **Step 6: Commit documentation and verification additions**

```bash
git add README.md worker/slop_detector/main.py tests/python/test_worker_cli.py
git commit -m "docs: document secure local detector setup"
```

## Final Manual Smoke Test

- [ ] With a token in `.secrets/huggingface.token`, accept the two gated-model pages in the same Hugging Face account.
- [ ] Run `kotlin slop-detector.main.kts tests/fixtures/article-with-image.webarchive --images --verbose`.
- [ ] Confirm the command uses MPS when available, runs all three text models in the stated order, names the EditLens score as estimated AI-edit extent, and shows Organika image output separately from metadata heuristics.
- [ ] Confirm `git status --short` does not list `.secrets/`, `.venv/`, or `.model-cache/`.
