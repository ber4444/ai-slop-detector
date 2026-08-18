# Markdown Slop Detector — Design

## Goal

Provide a personal, local-first command-line tool that examines Markdown files for likely AI-generated text. The command prints a readable report for a single file, or a corpus survey — how many files each detector flags, with a histogram — for many files at once.

## Scope

The initial release is for an Apple M4 Mac mini with 24 GB RAM. It uses local Hugging Face model weights after their first download and never sends file content to a remote inference service.

Text models, evaluated sequentially:

- `pangram/editlens_Llama-3.2-3B`
- `ogmatrixllm/glyph-v1.1`
- `ShantanuT01/vanguard-ai-text-detector`

The image path and the `.webarchive` reader from earlier revisions have been removed: Markdown references images rather than embedding them, so there was nothing on-disk to evaluate, and dropping both also drops the beautifulsoup4, Pillow, and defusedxml dependencies.

## Architecture

`slop-detector.main.kts` is the public Kotlin CLI. It validates arguments, creates or reuses the project-local Python virtual environment, and invokes a Python worker with a small structured request and response protocol.

The Python worker owns compatibility with Hugging Face Transformers, PEFT, PyTorch, and Apple MPS. It reduces each Markdown file to readable prose, chunks every file into one shared partition, and evaluates models one at a time to constrain memory use. It caches packages and model weights in gitignored project-local directories.

This hybrid boundary keeps the user-facing command Kotlin while using the supported local runtime for the requested model architectures. EditLens is loaded as the published PEFT adapter on `meta-llama/Llama-3.2-3B`.

## Command Interface

```text
./slop-detector.main.kts <file|glob|directory>... [--verbose]
```

- Inputs are Markdown (`.md`, `.markdown`, `.mdown`, `.mkd`). A glob is expanded by the tool when the shell has not, and a directory is walked recursively.
- One file yields a per-detector verdict. Several files yield a corpus survey: each file is scored on its own (the mean of its chunks), and the report states how many files each probability detector flags at the 0.50 threshold, a histogram of per-file scores, the flagged files by name, and the pooled aggregate as a footnote.
- Markdown is reduced to prose before scoring: fenced code, front matter, link targets, image references, table rules, and HTML are removed; heading, link, list, and quotation text is kept.
- `--verbose` adds each file's score, the per-chunk spread, Cohen's kappa between detectors, extraction counts, and full error text.

The normal output remains terminal-only and human-readable. It presents per-model scores, labels, and an interpretation reminder; it deliberately does not merge the scores into one aggregate percentage.

## Model Handling

The worker uses MPS when available and falls back to CPU with a visible warning. Models run sequentially and release references before the next model.

Text is split at natural boundaries into one shared set of model-safe chunks, sized for the narrowest token window among the detectors. Every detector scores that same partition, so per-model scores are means over identical units and can be compared chunk by chunk; the report states the chunk-level agreement between probability detectors as Cohen's kappa. In a corpus survey the chunks are grouped back by file to give each file its own score, and each probability detector reports how many files it flags at the 0.50 threshold.

All detector results are advisory and unsuitable as the sole basis for high-stakes decisions. EditLens reports an estimated AI-edit extent, a different quantity from the others' probability of AI authorship, so it is excluded from the flag count and the agreement statistic.

## Credentials and Downloads

The Hugging Face token is read from `.secrets/huggingface.token`; the complete `.secrets/` directory is ignored by Git. Setup documentation directs the user to accept Hugging Face access conditions for both `pangram/editlens_Llama-3.2-3B` and `meta-llama/Llama-3.2-3B` before first run. The token needs permission to read those accepted gated repositories.

First invocation bootstraps the Python environment and downloads required weights. Later executions use the local cache. An explicit offline mode is deferred; cache misses clearly state that a download is required.

## Failure Policy

- Unsupported or unreadable files fail before model startup, with a nonzero exit code.
- Missing Python, virtual-environment setup failures, and a missing token fail nonzero and name the cause before any model loads.
- An individual text model that cannot run — an inaccessible gated repository, a download failure, a model error — is reported as unavailable with its reason, and the detectors that did run are still reported. The run fails nonzero only when no text detector produced a score.
- An unavailable MPS device falls back to CPU and warns about slower execution.
- No file contents, scores, or tokens are logged by default.

## Test Strategy

Fast offline tests cover command parsing, Markdown-to-prose reduction, worker request/response validation, chunk boundaries and per-file aggregation, score/label formatting, the false-positive flag count, and the histogram rendering. A fake Python worker exercises the Kotlin CLI end to end.

The project also documents an opt-in smoke command for a real local model run. It is not part of the normal test suite because it needs a Hugging Face token, gated-model access, downloads, and substantial hardware time.

## Non-Goals

- Browser extension or URL fetching.
- Remote/API inference.
- Images, webarchives, or any non-Markdown input.
- A single cross-model AI-authorship verdict.
- Commercial use of models whose licenses prohibit it.
