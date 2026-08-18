from __future__ import annotations

import gc
import math
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .contracts import DetectorResult, ImageResult
from .metadata import inspect_image_metadata
from .webarchive import EmbeddedImage


MAX_CHUNK_CHARACTERS = 6000


@dataclass(frozen=True)
class TextModel:
    """A text detector plus the facts about it that its model card dictates."""

    name: str
    model_id: str
    label: str
    #: Chunk budget in characters, kept safely inside the model's token window
    #: so text is chunked rather than silently truncated.
    max_characters: int = MAX_CHUNK_CHARACTERS
    #: Output index meaning "machine written"; None derives it from id2label,
    #: which only works when the model publishes real label names.
    ai_index: int | None = None
    fast_tokenizer: bool = True


TEXT_MODELS = (
    # Glyph publishes placeholder labels; its card states LABEL_0 = human and
    # LABEL_1 = AI, and requires the slow tokenizer plus a 512-token window.
    TextModel(
        name="Glyph",
        model_id="ogmatrixllm/glyph-v1.1",
        label="probability AI-generated",
        max_characters=1600,
        ai_index=1,
        fast_tokenizer=False,
    ),
    # Vanguard is ModernBERT-large with a single sigmoid head emitting P(AI)
    # directly, over an 8192-token window.
    TextModel(
        name="Vanguard",
        model_id="ShantanuT01/vanguard-ai-text-detector",
        label="probability AI-generated",
    ),
)
EDITLENS_NAME = "EditLens"
EDITLENS_BASE = "meta-llama/Llama-3.2-3B"
EDITLENS_ADAPTER = "pangram/editlens_Llama-3.2-3B"
EDITLENS_LABEL = "estimated AI-edit extent"
EDITLENS_MAX_CHARACTERS = MAX_CHUNK_CHARACTERS
IMAGE_MODEL = "Organika/sdxl-detector"

NO_TEXT_DETAIL = "no readable text"

# Sentence endings and blank-line/paragraph breaks are the natural boundaries a
# chunk may be cut at before falling back to word and character splits.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def select_device(torch_module: object) -> str:
    """Return the accelerator to run on, preferring Apple MPS over the CPU."""
    try:
        available = bool(torch_module.backends.mps.is_available())
    except AttributeError:
        available = False
    return "mps" if available else "cpu"


def chunk_text(text: str, max_characters: int = MAX_CHUNK_CHARACTERS) -> list[str]:
    """Split text into model-safe chunks, preferring natural text boundaries."""
    if max_characters < 1:
        raise ValueError("max_characters must be at least 1")

    chunks: list[str] = []
    current = ""
    for segment in _BOUNDARY.split(text):
        for piece in _split_overlong(segment.strip(), max_characters):
            candidate = f"{current} {piece}" if current else piece
            if len(candidate) <= max_characters:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _split_overlong(segment: str, max_characters: int) -> list[str]:
    """Break a single segment that cannot fit, on words then on characters."""
    if not segment:
        return []
    if len(segment) <= max_characters:
        return [segment]

    pieces: list[str] = []
    current = ""
    for word in segment.split():
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_characters:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        while len(word) > max_characters:
            pieces.append(word[:max_characters])
            word = word[max_characters:]
        current = word
    if current:
        pieces.append(current)
    return pieces


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a plain list of logits."""
    if not logits:
        raise ValueError("softmax requires at least one logit")
    largest = max(logits)
    exponentials = [math.exp(value - largest) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def ai_probability(logits: list[float], ai_index: int | None) -> float:
    """Read P(AI) from one model's raw output.

    A single-logit head is a binary sigmoid classifier whose output already is
    P(AI); anything wider is a softmax over classes, one of which means AI.
    """
    if not logits:
        raise ValueError("a classifier must emit at least one logit")
    if len(logits) == 1:
        return 1.0 / (1.0 + math.exp(-logits[0]))
    if ai_index is None:
        raise ValueError("a multi-class classifier needs the AI label index")
    return softmax(logits)[ai_index]


def editlens_score(logits: list[float]) -> float:
    """Normalize EditLens bucket logits to an expected AI-edit extent in [0, 1]."""
    if len(logits) < 2:
        raise ValueError("EditLens requires at least two edit-extent buckets")
    probabilities = softmax(logits)
    expected = sum(index * value for index, value in enumerate(probabilities))
    return expected / (len(probabilities) - 1)


def mean_score(scores: list[float]) -> float:
    """Average chunk-level scores; no chunks means there is nothing to report."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


class ModelLoader(Protocol):
    """Loads one model at a time so only one set of weights is resident."""

    def load_text(self, model: TextModel, device: str) -> Callable[[str], dict]:
        """Return a predictor mapping a chunk to `{"ai": probability}`."""

    def load_editlens(
        self, base_id: str, adapter_id: str, device: str
    ) -> Callable[[str], list[float]]:
        """Return a predictor mapping a chunk to EditLens bucket logits."""

    def load_image(self, model_id: str, device: str) -> Callable[[bytes], float]:
        """Return a predictor mapping image bytes to an `artificial` probability."""


def run_text_detectors(
    text: str, loader: ModelLoader, device: str, verbose: bool = False
) -> list[DetectorResult]:
    """Run every text detector sequentially, keeping their meanings distinct.

    Each model is chunked to its own token window, so the chunk count can
    legitimately differ between detectors. A detector that cannot run is
    reported with no score and the reason, so the detectors that did run stay
    available.
    """
    if not chunk_text(text, MAX_CHUNK_CHARACTERS):
        return [
            DetectorResult(model.name, 0.0, model.label, NO_TEXT_DETAIL)
            for model in TEXT_MODELS
        ] + [DetectorResult(EDITLENS_NAME, 0.0, EDITLENS_LABEL, NO_TEXT_DETAIL)]

    results: list[DetectorResult] = []
    for model in TEXT_MODELS:
        results.append(
            _score_text_model(text, model, loader, device, verbose)
        )
    results.append(_score_editlens(text, loader, device))
    return results


def _score_text_model(
    text: str, model: TextModel, loader: ModelLoader, device: str, verbose: bool
) -> DetectorResult:
    chunks = chunk_text(text, model.max_characters)
    predictor = None
    try:
        predictor = loader.load_text(model, device)
        predictions = [predictor(chunk) for chunk in chunks]
    except Exception as error:  # noqa: BLE001 - one detector must not stop the rest
        return DetectorResult(model.name, None, model.label, _unavailable(error))
    finally:
        _release(loader, predictor)

    scores = [float(prediction["ai"]) for prediction in predictions]
    detail = _chunk_detail(chunks)
    if verbose:
        detail = "; ".join(filter(None, [detail, _raw_label_detail(predictions)]))
    return DetectorResult(model.name, mean_score(scores), model.label, detail)


def _score_editlens(text: str, loader: ModelLoader, device: str) -> DetectorResult:
    chunks = chunk_text(text, EDITLENS_MAX_CHARACTERS)
    predictor = None
    try:
        predictor = loader.load_editlens(EDITLENS_BASE, EDITLENS_ADAPTER, device)
        scores = [editlens_score(list(predictor(chunk))) for chunk in chunks]
    except Exception as error:  # noqa: BLE001 - one detector must not stop the rest
        return DetectorResult(EDITLENS_NAME, None, EDITLENS_LABEL, _unavailable(error))
    finally:
        _release(loader, predictor)

    return DetectorResult(
        EDITLENS_NAME, mean_score(scores), EDITLENS_LABEL, _chunk_detail(chunks)
    )


def _raw_label_detail(predictions: list[dict]) -> str:
    """Summarize the labels the model itself emitted, for --verbose auditing."""
    labels = [str(prediction["label"]) for prediction in predictions if prediction.get("label")]
    if not labels:
        return ""
    counted = sorted({label: labels.count(label) for label in labels}.items())
    return "raw labels " + ", ".join(f"{label}x{count}" for label, count in counted)


def run_image_detector(
    images: list[EmbeddedImage], loader: ModelLoader, device: str
) -> list[ImageResult]:
    """Score embedded images independently of any text result.

    A single unreadable image is skipped and reported rather than failing the
    run, and metadata heuristics stay separate from the model prediction.
    """
    if not images:
        return []

    try:
        predictor = loader.load_image(IMAGE_MODEL, device)
    except Exception as error:  # noqa: BLE001 - text results must stay available
        reason = _skip_reason(error)
        return [
            ImageResult(
                image.index, None, None, inspect_image_metadata(image.data), reason
            )
            for image in images
        ]

    results: list[ImageResult] = []
    for image in images:
        flags = inspect_image_metadata(image.data)
        try:
            score = float(predictor(image.data))
        except Exception as error:  # noqa: BLE001 - one bad image must not stop the run
            results.append(
                ImageResult(image.index, None, None, flags, _skip_reason(error))
            )
            continue
        label = "artificial" if score >= 0.5 else "human"
        results.append(ImageResult(image.index, score, label, flags, None))
    _release(loader, predictor)
    return results


def _chunk_detail(chunks: list[str]) -> str:
    return f"{len(chunks)} chunk" if len(chunks) == 1 else f"{len(chunks)} chunks"


def _skip_reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def _unavailable(error: Exception) -> str:
    """Explain why a detector could not run, keeping any guidance the error carries."""
    if isinstance(error, RuntimeError) and str(error).strip():
        return str(error).strip()
    return _skip_reason(error)


def _release(loader: ModelLoader, predictor: object) -> None:
    """Drop the predictor and let the loader free any accelerator memory."""
    if predictor is None:
        return
    del predictor
    release = getattr(loader, "release", None)
    if callable(release):
        release()
    gc.collect()


def select_runtime_device() -> tuple[str, list[str]]:
    """Choose the inference device and warn when falling back to the CPU."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is not installed in the worker environment: "
            "install worker/requirements.txt before running detectors"
        ) from error

    device = select_device(torch)
    if device == "cpu":
        return device, [
            "Apple MPS is unavailable, so models run on the CPU and will be "
            "substantially slower."
        ]
    return device, []


# Labels different detectors use for the machine-written class, and for the
# human-written class when only the complementary label can be matched.
_AI_LABEL_HINTS = ("ai", "artificial", "machine", "fake", "generated", "llm", "bot")
_HUMAN_LABEL_HINTS = ("human", "real", "natural", "authentic", "genuine")


def resolve_ai_label_index(id2label: dict[int, str], model_id: str) -> int:
    """Find which classifier output means "machine written" for this model.

    Detectors disagree about label order, so the index is derived from the
    model's own `id2label` rather than assumed.
    """
    labels = {int(index): str(name).lower() for index, name in id2label.items()}
    for index, name in sorted(labels.items()):
        if any(hint == name or hint in name.split("_") for hint in _AI_LABEL_HINTS):
            return index
    for index, name in sorted(labels.items()):
        if any(hint in name for hint in _AI_LABEL_HINTS):
            return index
    if len(labels) == 2:
        for index, name in sorted(labels.items()):
            if any(hint in name for hint in _HUMAN_LABEL_HINTS):
                return next(other for other in labels if other != index)
    raise RuntimeError(
        f"{model_id}: cannot tell which label means AI-generated from "
        f"{sorted(labels.values())}"
    )


GATED_ACCESS_HELP = (
    "Accept the access conditions for https://huggingface.co/"
    f"{EDITLENS_ADAPTER} and https://huggingface.co/{EDITLENS_BASE} with the "
    "same Hugging Face account as the token in .secrets/huggingface.token."
)


class TransformersLoader:
    """Loads Hugging Face weights one model at a time from the local cache."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._torch = None

    @property
    def torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def load_text(self, model: TextModel, device: str) -> Callable[[str], dict]:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = self._download(
            AutoTokenizer, model.model_id, use_fast=model.fast_tokenizer
        )
        classifier = self._download(
            AutoModelForSequenceClassification, model.model_id
        )
        # ModernBERT requests torch.compile, which has no MPS path. Declining it
        # before the first forward keeps the model from warning about it; the
        # compiled path was never available on Metal anyway.
        if device != "cuda" and getattr(classifier.config, "reference_compile", False):
            classifier.config.reference_compile = False
        classifier.eval().to(device)
        limit = _token_limit(tokenizer, classifier.config)
        ai_index = self._ai_index(model, classifier.config)

        def predict(chunk: str) -> dict:
            inputs = tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=limit
            ).to(device)
            with self.torch.inference_mode():
                logits = classifier(**inputs).logits[0]
            values = logits.float().tolist()
            return {
                "ai": ai_probability(values, ai_index),
                "label": _raw_label(values, ai_index, classifier.config),
            }

        return predict

    def _ai_index(self, model: TextModel, config: object) -> int | None:
        """Decide which output means AI, trusting the model card over id2label."""
        if getattr(config, "num_labels", 2) == 1:
            return None  # single sigmoid head: the logit already means P(AI)
        if model.ai_index is not None:
            return model.ai_index
        return resolve_ai_label_index(config.id2label, model.model_id)

    def load_editlens(
        self, base_id: str, adapter_id: str, device: str
    ) -> Callable[[str], list[float]]:
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        labels = self._editlens_bucket_count(adapter_id)
        tokenizer = self._download(AutoTokenizer, adapter_id, fallback_id=base_id)
        base = self._download(
            AutoModelForSequenceClassification,
            base_id,
            num_labels=labels,
            dtype=self.torch.float16 if device == "mps" else self.torch.float32,
        )
        if base.config.pad_token_id is None:
            base.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = self._attach_adapter(PeftModel, base, adapter_id)
        model.eval().to(device)
        limit = _token_limit(tokenizer, base.config)

        def predict(chunk: str) -> list[float]:
            inputs = tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=limit
            ).to(device)
            with self.torch.inference_mode():
                logits = model(**inputs).logits[0]
            return [float(value) for value in logits.float().tolist()]

        return predict

    def _editlens_bucket_count(self, adapter_id: str) -> int:
        """Read the number of edit-extent buckets from the adapter's own head.

        The adapter repository ships no `config.json`, so the bucket count comes
        from the shape of the classification head saved with the adapter.
        """
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        try:
            weights = hf_hub_download(
                adapter_id, "adapter_model.safetensors", token=self._token
            )
        except Exception as error:  # noqa: BLE001 - reported with actionable guidance
            raise RuntimeError(_download_failure(adapter_id, error)) from error

        with safe_open(weights, framework="pt") as adapter:
            for key in adapter.keys():
                if key.endswith(("score.weight", "classifier.weight")):
                    return int(adapter.get_slice(key).get_shape()[0])
        raise RuntimeError(
            f"{adapter_id}: no classification head found in the adapter, so its "
            "edit-extent buckets cannot be counted"
        )

    def load_image(self, model_id: str, device: str) -> Callable[[bytes], float]:
        from io import BytesIO

        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        # Organika saved a slow processor; asking for it explicitly keeps the
        # preprocessing the model was trained with, and stays quiet about it.
        processor = self._download(AutoImageProcessor, model_id, use_fast=False)
        model = self._download(AutoModelForImageClassification, model_id)
        model.eval().to(device)
        ai_index = resolve_ai_label_index(model.config.id2label, model_id)

        def predict(data: bytes) -> float:
            with Image.open(BytesIO(data)) as image:
                # PIL always hands back channels-last pixels; saying so keeps the
                # processor from guessing wrong on very small images.
                pixels = processor(
                    images=image.convert("RGB"),
                    return_tensors="pt",
                    input_data_format="channels_last",
                )
            pixels = pixels.to(device)
            with self.torch.inference_mode():
                logits = model(**pixels).logits[0]
            return self.torch.softmax(logits.float(), dim=-1)[ai_index].item()

        return predict

    def release(self) -> None:
        """Free accelerator memory held by the model that just finished."""
        gc.collect()
        empty_cache = getattr(getattr(self.torch, "mps", None), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def _download(self, factory, model_id: str, fallback_id: str | None = None, **kwargs):
        try:
            return factory.from_pretrained(model_id, token=self._token, **kwargs)
        except Exception as error:  # noqa: BLE001 - reported with actionable guidance
            if fallback_id is not None:
                return self._download(factory, fallback_id, **kwargs)
            raise RuntimeError(_download_failure(model_id, error)) from error

    def _attach_adapter(self, peft_model, base, adapter_id: str):
        try:
            return peft_model.from_pretrained(base, adapter_id, token=self._token)
        except Exception as error:  # noqa: BLE001 - reported with actionable guidance
            raise RuntimeError(_download_failure(adapter_id, error)) from error


def _download_failure(model_id: str, error: Exception) -> str:
    message = f"{model_id}: {type(error).__name__}: {error}"
    if _looks_gated(error):
        return f"{message}\n{GATED_ACCESS_HELP}"
    return message


def _looks_gated(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(
        hint in text
        for hint in ("gated", "401", "403", "awaiting a review", "access to model")
    )


def _raw_label(logits: list[float], ai_index: int | None, config: object) -> str:
    """The label the model itself would report, shown under --verbose."""
    if len(logits) == 1:
        threshold = 0.5
        return "ai" if ai_probability(logits, ai_index) >= threshold else "human"
    winner = max(range(len(logits)), key=lambda index: logits[index])
    return str(getattr(config, "id2label", {}).get(winner, winner))


def _token_limit(tokenizer: object, config: object) -> int:
    """Pick a truncation length the model can actually accept."""
    candidates = [
        getattr(tokenizer, "model_max_length", None),
        getattr(config, "max_position_embeddings", None),
    ]
    lengths = [value for value in candidates if isinstance(value, int) and 0 < value < 100_000]
    return min(lengths) if lengths else 512
