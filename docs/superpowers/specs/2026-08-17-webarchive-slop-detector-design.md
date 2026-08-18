# Webarchive Slop Detector — Design

## Goal

Provide a personal, local-first command-line tool that examines a macOS `.webarchive` file for likely AI-generated text and, on request, likely AI-generated images. The command prints a readable report suitable for reviewing one archived web page at a time.

## Scope

The initial release is for an Apple M4 Mac mini with 24 GB RAM. It uses local Hugging Face model weights after their first download and never sends archive content to a remote inference service.

Text models, evaluated sequentially:

- `pangram/editlens_Llama-3.2-3B`
- `ogmatrixllm/glyph-v1.1`
- `ShantanuT01/vanguard-ai-text-detector`

The optional image path (`--images`) evaluates embedded raster images with `Organika/sdxl-detector` and reports metadata heuristics separately.

## Architecture

`slop-detector.main.kts` is the public Kotlin CLI. It validates arguments, creates or reuses the project-local Python virtual environment, and invokes a Python worker with a small structured request and response protocol.

The Python worker owns compatibility with Hugging Face Transformers, PEFT, PyTorch, and Apple MPS. It extracts the `.webarchive` property-list payload, identifies readable HTML text, and evaluates models one at a time to constrain memory use. It caches packages and model weights in gitignored project-local directories.

This hybrid boundary keeps the user-facing command Kotlin while using the supported local runtime for the requested model architectures. EditLens is loaded as the published PEFT adapter on `meta-llama/Llama-3.2-3B`.

## Command Interface

```text
./slop-detector.main.kts <archive.webarchive> [--images] [--verbose]
```

- The default path extracts and analyzes readable text only.
- `--images` enables embedded-image extraction, the image detector, and image metadata inspection.
- `--verbose` adds extraction counts, chunk counts, raw detector labels, and skip reasons.

The normal output remains terminal-only and human-readable. It presents per-model scores, labels, and an interpretation reminder; it deliberately does not merge the scores into one aggregate percentage.

## Model Handling

The worker uses MPS when available and falls back to CPU with a visible warning. Models run sequentially and release references before the next model.

Long article text is split at natural text boundaries into model-safe chunks. The report includes the model-level results and explains how its final text score is derived from chunks. Images are evaluated independently; their artificial/human result is never added to a text score.

Metadata heuristics inspect available EXIF, XMP, PNG text chunks, and similar container metadata for generator/software fields. A missing field is neutral rather than evidence of human origin. Metadata findings are labeled as heuristics, not detector predictions.

The Organika model specifically detects SDXL-like generated imagery, so the report warns that its confidence is not a universal AI-image provenance determination. All detector results are advisory and unsuitable as the sole basis for high-stakes decisions.

## Credentials and Downloads

The Hugging Face token is read from `.secrets/huggingface.token`; the complete `.secrets/` directory is ignored by Git. Setup documentation directs the user to accept Hugging Face access conditions for both `pangram/editlens_Llama-3.2-3B` and `meta-llama/Llama-3.2-3B` before first run. The token needs permission to read those accepted gated repositories.

First invocation bootstraps the Python environment and downloads required weights. Later executions use the local cache. An explicit offline mode is deferred; cache misses clearly state that a download is required.

## Failure Policy

- Invalid or unreadable archives fail before model startup, with a nonzero exit code.
- Missing Python, virtual-environment setup failures, missing token, inaccessible gated repositories, and text-model errors fail nonzero and name the cause/model.
- If `--images` is used, a malformed or unsupported individual image is skipped and shown in the report; successful text results remain available.
- An unavailable MPS device falls back to CPU and warns about slower execution.
- No archive contents, scores, tokens, or image bytes are logged by default.

## Test Strategy

Fast offline tests cover command parsing, archive-content selection, worker request/response validation, chunk boundaries and aggregation, score/label formatting, and metadata heuristics. Fixture `.webarchive` files exercise ordinary HTML, malformed data, multiple embedded images, and absent metadata.

The project also documents an opt-in smoke command for a real local model run. It is not part of the normal test suite because it needs a Hugging Face token, gated-model access, downloads, and substantial hardware time.

## Non-Goals

- Browser extension or URL fetching.
- Remote/API inference.
- OCR of image contents.
- A single cross-model AI-authorship verdict.
- Commercial use of models whose licenses prohibit it.
