# Webarchive Slop Detector

A personal, local-first command that examines one macOS `.webarchive` file for
likely AI-generated text and, on request, likely AI-generated images. Everything
runs on your own machine: after the first model download, no archive content
leaves the computer and no remote inference API is used.

```bash
kotlin slop-detector.main.kts saved-page.webarchive
```

## What it reports

- **Text detectors** — `ogmatrixllm/glyph-v1.1`, `ShantanuT01/vanguard-ai-text-detector`,
  and `pangram/editlens_Llama-3.2-3B` (a PEFT adapter on `meta-llama/Llama-3.2-3B`),
  each run separately over the readable article text.
- **Image detectors** (`--images`) — `Organika/sdxl-detector` over the raster
  images embedded in the archive.
- **Metadata heuristics** (`--images`) — generator/software fields found in EXIF,
  XMP, or PNG text chunks.

Scores are deliberately **never merged into a single AI percentage**. Each model
means something different, and image results are never folded into a text score.

## Requirements

- macOS on Apple silicon (developed against an M4 Mac mini with 24 GB RAM). MPS
  is used when available; otherwise the run falls back to the CPU and warns that
  it will be substantially slower.
- Kotlin (`brew install kotlin`).
- Python 3.11 or newer available as `python3`.
- Roughly 8 GB of free disk for the downloaded weights.

## Hugging Face access

Put a read token in `.secrets/huggingface.token` (one token only, no `HF_TOKEN=`
prefix, no quotes). The whole `.secrets/` directory is excluded from Git and the
token value is never printed, not even in error messages.

Before the first run, sign in to that same Hugging Face account and accept the
access conditions for both gated repositories:

- <https://huggingface.co/pangram/editlens_Llama-3.2-3B>
- <https://huggingface.co/meta-llama/Llama-3.2-3B>

The token needs read permission for accepted gated repositories. If access is
missing, the run stops with a nonzero exit code and names both pages.

## First run

The Kotlin script bootstraps everything it needs before starting the worker:

1. Creates `.venv` with `python3 -m venv .venv` if it does not exist.
2. Installs `worker/requirements.txt` (re-installed whenever that file changes).
3. Points `HF_HOME` at `.model-cache/` inside the repository.
4. Downloads any missing model weights.

`.venv/`, `.model-cache/`, and `.secrets/` are gitignored. Later runs reuse the
local cache. There is no offline mode yet: a cache miss states plainly that a
download is required.

## Usage

```bash
kotlin slop-detector.main.kts saved-page.webarchive
```

```bash
kotlin slop-detector.main.kts saved-page.webarchive --images --verbose
```

- default — extracts and analyzes readable text only.
- `--images` — also extracts embedded raster images, runs the image detector,
  and inspects image metadata.
- `--verbose` — adds the device, extraction counts, chunk counts, and skip
  reasons to the notes.

Exit codes: `0` success, `2` invalid input (bad arguments or an unreadable
archive, detected before any model loads), `1` setup or model failure (missing
Python environment, missing token, inaccessible gated repository, model error).
With `--images`, a single malformed image is skipped and shown in the report
rather than failing the run.

## How to read the output

- Every score is **advisory**. AI-text detection is unreliable enough that no
  result here should be the sole basis for a consequential decision, and none of
  it is a provenance assertion.
- **EditLens** reports an *estimated AI-edit extent*, not a probability that the
  page was written by a model.
- **Glyph** and **Vanguard** report their own probability that the text is
  machine-written; they can and will disagree.
- Long text is split at sentence and paragraph boundaries into model-safe
  chunks, and each model's reported score is the mean over its chunks. The chunk
  count is printed next to each score.
- **Organika/sdxl-detector** specifically recognizes SDXL-like generated
  imagery. A low score is not a universal "this image is real" determination.
- **Metadata heuristics** are container heuristics, not detector predictions. A
  missing generator field is neutral — plenty of pipelines strip metadata.

## Tests

The offline suite needs no token, no network, and no model weights:

```bash
.venv/bin/python -m pip install pytest Pillow beautifulsoup4 defusedxml
```

```bash
.venv/bin/python -m pytest -q
```

`tests/kotlin/test_cli.py` runs the real Kotlin script against a fake Python
worker; it is skipped automatically when Kotlin is not installed.

### Opt-in smoke test

This one is *not* part of the normal suite: it needs a token, accepted gated
models, downloads, and substantial hardware time.

```bash
kotlin slop-detector.main.kts tests/fixtures/article-with-image.webarchive --images
```

## Licenses and intended use

EditLens is CC BY-NC-SA 4.0 and Organika's SDXL detector is CC BY-NC 3.0, so
this repository's intended use is personal and non-commercial. Llama 3.2 is
covered by the Llama 3.2 Community License. Check each model card before using
any of this for anything else.

## Non-goals

No browser extension, no URL fetching, no remote inference, no OCR of image
contents, and no single cross-model AI-authorship verdict.
