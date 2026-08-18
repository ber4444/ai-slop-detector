from __future__ import annotations

import gc
import math
import re
from typing import Callable, Protocol

from .contracts import DetectorResult, ImageResult
from .metadata import inspect_image_metadata
from .webarchive import EmbeddedImage


MAX_CHUNK_CHARACTERS = 6000

TEXT_MODELS = (
    ("Glyph", "ogmatrixllm/glyph-v1.1", "likely AI-generated"),
    ("Vanguard", "ShantanuT01/vanguard-ai-text-detector", "likely AI-generated"),
)
EDITLENS_NAME = "EditLens"
EDITLENS_BASE = "meta-llama/Llama-3.2-3B"
EDITLENS_ADAPTER = "pangram/editlens_Llama-3.2-3B"
EDITLENS_LABEL = "estimated AI-edit extent"
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

    def load_text(self, model_id: str, device: str) -> Callable[[str], dict]:
        """Return a predictor mapping a chunk to `{"ai": probability}`."""

    def load_editlens(
        self, base_id: str, adapter_id: str, device: str
    ) -> Callable[[str], list[float]]:
        """Return a predictor mapping a chunk to EditLens bucket logits."""

    def load_image(self, model_id: str, device: str) -> Callable[[bytes], float]:
        """Return a predictor mapping image bytes to an `artificial` probability."""


def run_text_detectors(
    text: str, loader: ModelLoader, device: str
) -> list[DetectorResult]:
    """Run every text detector sequentially, keeping their meanings distinct."""
    chunks = chunk_text(text)
    if not chunks:
        return [
            DetectorResult(name, 0.0, label, NO_TEXT_DETAIL)
            for name, _, label in TEXT_MODELS
        ] + [DetectorResult(EDITLENS_NAME, 0.0, EDITLENS_LABEL, NO_TEXT_DETAIL)]

    detail = _chunk_detail(chunks)
    results: list[DetectorResult] = []
    for name, model_id, label in TEXT_MODELS:
        predictor = loader.load_text(model_id, device)
        scores = [float(predictor(chunk)["ai"]) for chunk in chunks]
        results.append(DetectorResult(name, mean_score(scores), label, detail))
        _release(loader, predictor)

    predictor = loader.load_editlens(EDITLENS_BASE, EDITLENS_ADAPTER, device)
    scores = [editlens_score(list(predictor(chunk))) for chunk in chunks]
    results.append(
        DetectorResult(EDITLENS_NAME, mean_score(scores), EDITLENS_LABEL, detail)
    )
    _release(loader, predictor)
    return results


def run_image_detector(
    images: list[EmbeddedImage], loader: ModelLoader, device: str
) -> list[ImageResult]:
    """Score embedded images independently of any text result.

    A single unreadable image is skipped and reported rather than failing the
    run, and metadata heuristics stay separate from the model prediction.
    """
    if not images:
        return []

    predictor = loader.load_image(IMAGE_MODEL, device)
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


def _release(loader: ModelLoader, predictor: object) -> None:
    """Drop the predictor and let the loader free any accelerator memory."""
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

    def load_text(self, model_id: str, device: str) -> Callable[[str], dict]:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = self._download(AutoTokenizer, model_id)
        model = self._download(AutoModelForSequenceClassification, model_id)
        model.eval().to(device)
        ai_index = resolve_ai_label_index(model.config.id2label, model_id)
        limit = _token_limit(tokenizer, model.config)

        def predict(chunk: str) -> dict:
            inputs = tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=limit
            ).to(device)
            with self.torch.inference_mode():
                logits = model(**inputs).logits[0]
            probabilities = self.torch.softmax(logits.float(), dim=-1)
            return {"ai": probabilities[ai_index].item()}

        return predict

    def load_editlens(
        self, base_id: str, adapter_id: str, device: str
    ) -> Callable[[str], list[float]]:
        from peft import PeftModel
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        adapter_config = self._download(AutoConfig, adapter_id)
        labels = getattr(adapter_config, "num_labels", None)
        if not labels or labels < 2:
            raise RuntimeError(
                f"{adapter_id}: the adapter does not declare its edit-extent "
                "buckets, so its score cannot be normalized"
            )

        tokenizer = self._download(AutoTokenizer, adapter_id, fallback_id=base_id)
        base = self._download(
            AutoModelForSequenceClassification,
            base_id,
            num_labels=labels,
            torch_dtype=self.torch.float16 if device == "mps" else self.torch.float32,
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

    def load_image(self, model_id: str, device: str) -> Callable[[bytes], float]:
        from io import BytesIO

        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        processor = self._download(AutoImageProcessor, model_id)
        model = self._download(AutoModelForImageClassification, model_id)
        model.eval().to(device)
        ai_index = resolve_ai_label_index(model.config.id2label, model_id)

        def predict(data: bytes) -> float:
            with Image.open(BytesIO(data)) as image:
                pixels = processor(images=image.convert("RGB"), return_tensors="pt")
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


def _token_limit(tokenizer: object, config: object) -> int:
    """Pick a truncation length the model can actually accept."""
    candidates = [
        getattr(tokenizer, "model_max_length", None),
        getattr(config, "max_position_embeddings", None),
    ]
    lengths = [value for value in candidates if isinstance(value, int) and 0 < value < 100_000]
    return min(lengths) if lengths else 512
