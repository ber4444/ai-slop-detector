# Markdown Slop Detector

A personal, local-first command that examines Markdown files for likely
AI-generated text. Point it at one file for a verdict, or at a whole directory
to survey a corpus and see **how many files each detector flags**. Everything
runs on your own machine: after the first model download, no content leaves the
computer and no remote inference API is used.

```text
% kotlin slop-detector.main.kts ~/KEEP/proposals/*.md
Surveyed 92 files, each scored on its own

Glyph flags 31 of 92 files as likely AI-generated
    0–10%    █████████████            11
    10–20%   ███████                  6
    20–30%   █████████████            11
    30–40%   █████████████████        14
    40–50%   ████████████████████████ 19
    50–60%   █████████████            11
    60–70%   ███████                  6
    70–80%   ████████                 7
    80–90%   ███                      3
    90–100%  █████                    4
  files it flags:
    94.2%  KEEP-0150-jvm-static-annotation-in-interface-companion.md
    ...
  (aggregate over all text pooled: 39.6% — footnote only)

Vanguard flags 0 of 92 files as likely AI-generated
  (aggregate over all text pooled: 9.6% — footnote only)
```

Those 92 files are the Kotlin language proposals — human-authored formal
technical prose. So every file a detector flags is a **false positive**, and
this run is a measured false-positive rate: Glyph 33.7%, Vanguard 0%, on the
same corpus. That gap is the point of running detectors side by side.

## Surveying a corpus

Give it several files, a glob, or a directory and each file is scored **on its
own**. The report shows, per detector:

- **how many files it flags** — a file counts as flagged when its own mean score
  is 0.50 or higher;
- **a histogram** of the per-file scores, so you see the whole distribution, not
  just the count;
- **the flagged files**, named with their scores, so you can go look;
- **the pooled aggregate** over all text, demoted to a one-line footnote — it is
  the least useful number here, because one long file can dominate it.

`--verbose` additionally lists every file's score, highest first.

A single file instead gets a plain verdict, with no histogram.

### What gets read from Markdown

Only prose. Fenced code blocks, YAML front matter, link targets, image
references, table rules, and HTML are removed before scoring; heading text, link
text, list items, and quotations are kept. Scoring a code block would measure
the writer's toolchain rather than their writing — and on a technical corpus
that is most of the document.

## The detectors

- `ogmatrixllm/glyph-v1.1` and `ShantanuT01/vanguard-ai-text-detector` each
  report their own probability that the text is machine-written.
- `pangram/editlens_Llama-3.2-3B` (a PEFT adapter on `meta-llama/Llama-3.2-3B`)
  reports an *estimated AI-edit extent*, a different quantity, so it is kept out
  of the flag count and the agreement statistic.

Scores are deliberately **never merged into a single number**. Each model means
something different, and they often disagree — the survey above is Glyph and
Vanguard reaching opposite conclusions on the same corpus.

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

EditLens also requires manual approval from its authors, which may be declined —
in that case EditLens is reported as `could not run` and the other detectors
still work.

Use a **read-only** token — this tool only downloads weights and never writes to
the Hub. A fine-grained token scoped to the model repositories is preferable to
a classic Read token, which can read everything your account can reach.
Fine-grained tokens also need the permission covering read access to public
gated repositories you have been granted.

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
kotlin slop-detector.main.kts notes.md
```

```bash
kotlin slop-detector.main.kts docs/*.md
```

```bash
kotlin slop-detector.main.kts 'docs/*.md'
```

```bash
kotlin slop-detector.main.kts docs --verbose
```

Inputs are Markdown (`.md`, `.markdown`, `.mdown`, `.mkd`). A glob works whether
or not your shell expands it — quoted globs are expanded by the tool. A directory
is walked recursively for supported files.

- `--verbose` — adds numbers to the plain-language report: each model's
  percentage, the per-chunk spread, every file's score, Cohen's kappa between
  detectors, the device, and full error text.

Exit codes: `0` success, `2` invalid input (bad arguments or an unsupported
file, detected before any model loads), `1` setup failure (missing Python
environment, missing token) or a run in which no detector could produce a score.

### Partial reports

One detector failing does not throw away the others. A model that cannot run — a
gated repository you have not been granted, a download failure, a model error —
is printed as `could not run` with its reason, the rest keep their results, and
a note names the gap. Such a run still exits `0`, so read the notes rather than
the exit code to know whether every detector ran.

## How to read the output

By default the report uses plain words, because a percentage is not something
most readers can act on. The wording describes how strongly a detector leans —
`almost certainly human` under 10%, `probably human` under 40%, `unclear` in the
middle, `probably AI-generated` from 60%, `almost certainly AI-generated` from
90%. It never says whether a detector is *right*: the tool has no ground truth to
check itself against, so a confident detector and a correct one look identical.

- Every score is **advisory**. AI-text detection is unreliable enough that no
  result here should be the sole basis for a consequential decision.
- Neither model reads 0% on human text. Formulaic, heavily edited prose —
  specifications, corporate blogs, academic abstracts — scores higher than
  casual writing. The KEEP survey above is exactly this: real human proposals,
  and Glyph still flags a third of them.
- Every detector scores one shared set of chunks, so their per-chunk scores are
  directly comparable. When two probability detectors both run, the report says
  whether they back each other up — in plain words by default, and as Cohen's
  kappa under `--verbose`. When one detector calls every chunk the same way,
  kappa is 0 by construction and the report says so instead of reading meaning
  into it.

## Tests

The offline suite needs no token, no network, and no model weights:

```bash
.venv/bin/python -m pip install pytest
```

```bash
.venv/bin/python -m pytest -q
```

`tests/kotlin/test_cli.py` runs the real Kotlin script against a fake Python
worker; it is skipped automatically when Kotlin is not installed.

### Polarity probe

Nothing in the normal output can tell you whether a detector's score is the
right way round — a reversed label mapping produces confident, plausible,
inverted numbers. This diagnostic settles direction against text whose
authorship is known: prose from packages released years before consumer LLMs,
and text generated by a language model.

```bash
HF_HOME=.model-cache .venv/bin/python tools/polarity_probe.py
```

Last run: Glyph separates known-machine from known-human by +0.53 mean score,
Vanguard by +1.00. Both are the right way round. Vanguard separates far more
sharply, which is consistent with its 0% false-positive rate on the KEEP corpus.

## Licenses and intended use

The code here is MIT — see [LICENSE](LICENSE). That covers this repository's
source only.

It does not cover the models, and it cannot: no weights are stored in or
distributed by this repository. Each is downloaded from Hugging Face under its
own terms, and one of them forbids commercial use outright. Full detail in
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).

| Model | License |
| --- | --- |
| `pangram/editlens_Llama-3.2-3B` | CC BY-NC-SA 4.0 — **non-commercial** |
| `meta-llama/Llama-3.2-3B` | Llama 3.2 Community License |
| `ShantanuT01/vanguard-ai-text-detector` | MIT |
| `ogmatrixllm/glyph-v1.1` | see its model card |

So while the code is permissively licensed, running this tool as configured is a
personal, non-commercial activity. Check each model card before relying on any
of it for anything else.

## Non-goals

No browser extension, no URL fetching, no remote inference, no images, and no
single cross-model AI-authorship verdict.
