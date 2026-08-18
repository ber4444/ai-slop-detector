# Webarchive Slop Detector

A personal, local-first command that examines saved web pages
(macOS `.webarchive`) and Markdown files for likely AI-generated text and, on
request, likely AI-generated images. Everything runs on your own machine: after
the first model download, no content leaves the computer and no remote
inference API is used.

```bash
kotlin slop-detector.main.kts saved-page.webarchive
```

```bash
kotlin slop-detector.main.kts docs/*.md
```

## Scanning several files

Give it more than one file and you get **one collective answer**, not one per
file. Every file is split into sections, all the sections are pooled, and each
detector's percentage is the mean over that pool. Sections never span a file
boundary, so each one is still attributable.

That means a long file weighs more than a short one, which is usually what you
want when asking "is this documentation set full of slop?" — and it means one
bad file can move the answer. `--verbose` breaks it back down:

```text
Read: 2 files, scored together as one answer

What the text detectors think
  Glyph: 17.8% — probably human (13 chunks; per-chunk 0.01-0.92, median 0.02, 2/13 over 0.50)

Notes:
  - Glyph per file: plan.md 12%, design.md 31%
```

### What gets read from Markdown

Only prose. Fenced code blocks, YAML front matter, link targets, image
references, table rules, and HTML are removed before scoring; heading text,
link text, list items, and quotations are kept. Scoring a code block would
measure the writer's toolchain rather than their writing.

Markdown references images rather than embedding them, so `--images` has
nothing to evaluate in a Markdown file. It still works for `.webarchive`
inputs, including when both kinds are scanned together.

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

Use a **read-only** token — this tool only downloads weights and never writes to
the Hub, so a write token would grant capability it never uses. A fine-grained
token scoped to the five model repositories is preferable to a classic Read
token, which can read everything your account can reach. Fine-grained tokens
also need the permission covering read access to public gated repositories you
have been granted; without it, EditLens and Llama fail with a 403 even after you
have accepted their licenses.

If access is missing, EditLens is reported as unavailable with both gated pages
named, and the other detectors still run — see [Partial reports](#partial-reports).

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
kotlin slop-detector.main.kts saved-page.webarchive --images --verbose
```

Inputs may be `.webarchive` or Markdown (`.md`, `.markdown`, `.mdown`, `.mkd`),
in any combination:

```bash
kotlin slop-detector.main.kts notes.md
```

```bash
kotlin slop-detector.main.kts docs/*.md
```

```bash
kotlin slop-detector.main.kts 'docs/*.md'
```

```bash
kotlin slop-detector.main.kts docs
```

A glob works whether or not your shell expands it — quoted globs are expanded
by the tool. A directory is walked recursively for supported files.

- default — extracts and analyzes readable text only.
- `--images` — also extracts embedded raster images, runs the image detector,
  and inspects image metadata.
- `--verbose` — switches the report from plain language to numbers: each
  model's percentage, the per-chunk spread behind it, Cohen's kappa between
  detectors, the device, extraction counts, and full error text.

Exit codes: `0` success, `2` invalid input (bad arguments or an unreadable
archive, detected before any model loads), `1` setup failure (missing Python
environment, missing token) or a run in which no detector could produce a score.

### Partial reports

One detector failing does not throw away the others. A text model that cannot
run — a gated repository you have not been granted, a download failure, a model
error — is printed as `could not run` with its reason, the remaining detectors
keep their answers, and a note names the gap:

```text
What the text detectors think
  Glyph: probably AI-generated
  Vanguard: almost certainly human
  EditLens: could not run
      Accept the access conditions for https://huggingface.co/... 

Notes:
  - Not every detector ran: EditLens could not start. The results shown are
    unaffected by that.
```

Without `--verbose` the failure shows only its actionable last line; with it you
get the full error including request IDs and URLs.

Such a run still exits `0`, so read the notes rather than the exit code to know
whether every detector ran. With `--images`, a single malformed image is skipped
and shown; if the image model itself cannot load, every image is skipped and the
text results are unaffected.

## How to read the output

By default the report says what each detector thinks in plain words, because a
percentage is not something most readers can act on:

```text
What the text detectors think
  Glyph: probably AI-generated
  Vanguard: almost certainly human

Notes:
  - Glyph and Vanguard flatly contradict each other about this page: one reads
    it as AI-generated and the other as human. When they disagree this sharply,
    neither reading is trustworthy here. Treat the page as unresolved.
```

The wording describes how strongly a detector leans — `almost certainly human`
under 10%, `probably human` under 40%, `unclear` in the middle, `probably
AI-generated` from 60%, `almost certainly AI-generated` from 90%. It never says
whether a detector is *right*: the tool has no ground truth to check itself
against, so a confident detector and a correct one look identical here.

Pass `--verbose` for the numbers behind that wording.

- Every score is **advisory**. AI-text detection is unreliable enough that no
  result here should be the sole basis for a consequential decision, and none of
  it is a provenance assertion.
- **EditLens** reports an *estimated AI-edit extent*, not a probability that the
  page was written by a model.
- **Glyph** and **Vanguard** report their own probability that the text is
  machine-written; they can and will disagree.
- Both model cards document a 0.5 decision threshold. The wording is
  deliberately more reluctant than that: it will not say `almost certainly
  AI-generated` below 90%, and it says `unclear — could be either` between 40%
  and 60% rather than pretending 0.51 differs meaningfully from 0.49.
- Neither model reads 0% on human text. Formulaic, heavily edited prose —
  corporate blogs, press releases, academic abstracts — scores higher than
  casual writing while still landing well under the threshold. Glyph's own card
  reports 90.8% accuracy on arXiv abstracts against 100% on personal blogs, so
  a human-written corporate post reading 20–35% is normal.
- Long text is split at sentence and paragraph boundaries into one shared set
  of chunks that **every** detector scores, so their numbers are means of the
  same thing and can be compared chunk by chunk. `--verbose` prints the range,
  median, and how many chunks cleared 0.50 — a mean of 0.86 over three chunks
  can hide two saturated chunks and one near-zero.
- When two probability detectors both run, the report says whether they back
  each other up — in plain words by default, and as Cohen's kappa over the
  shared chunks under `--verbose`. This matters: two models disagreeing and one
  model being misread produce identical-looking averages, and only per-chunk
  comparison separates them. When one detector calls every chunk the same way,
  kappa is 0 by construction and the report says so instead of reading meaning
  into it.
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

### Polarity probe

Nothing in the normal output can tell you whether a detector's score is the
right way round — a reversed label mapping produces confident, plausible,
inverted numbers, and semantic labels like `human`/`ai` look authoritative
whether or not they were ever verified. This diagnostic settles direction
against text whose authorship is known: prose from packages released years
before consumer LLMs, and text generated by a language model.

```bash
HF_HOME=.model-cache .venv/bin/python tools/polarity_probe.py
```

Last run: Glyph separates known-machine from known-human by +0.53 mean score,
Vanguard by +1.00. Both are the right way round. Vanguard separates far more
sharply and its outputs saturate near 0.000 and 1.000, so an intermediate
Vanguard score deserves more suspicion than an intermediate Glyph score.

### Opt-in smoke test

This one is *not* part of the normal suite: it needs a token, accepted gated
models, downloads, and substantial hardware time.

```bash
kotlin slop-detector.main.kts tests/fixtures/article-with-image.webarchive --images
```

## Licenses and intended use

The code here is MIT — see [LICENSE](LICENSE). That covers this repository's
source only.

It does not cover the models, and it cannot: no weights are stored in or
distributed by this repository. Each is downloaded from Hugging Face under its
own terms, and two of them forbid commercial use outright. Full detail in
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).

| Model | License |
| --- | --- |
| `pangram/editlens_Llama-3.2-3B` | CC BY-NC-SA 4.0 — **non-commercial** |
| `Organika/sdxl-detector` | CC BY-NC 3.0 — **non-commercial** |
| `meta-llama/Llama-3.2-3B` | Llama 3.2 Community License |
| `ShantanuT01/vanguard-ai-text-detector` | MIT |
| `ogmatrixllm/glyph-v1.1` | see its model card |

So while the code is permissively licensed, running this tool as configured is
a personal, non-commercial activity. Check each model card before relying on
any of it for anything else.

## Non-goals

No browser extension, no URL fetching, no remote inference, no OCR of image
contents, and no single cross-model AI-authorship verdict.
