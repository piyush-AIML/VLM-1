"""
VLMCollator — production-grade data collator for Vision-Language Models.

Handles image preprocessing, text tokenization, and label masking
for encoder-decoder or decoder-only VLM training pipelines.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor

logger = logging.getLogger(__name__)

# Sentinel value used by HuggingFace to ignore positions in cross-entropy loss
_LABEL_IGNORE_INDEX = -100

# Required keys in each batch sample
_REQUIRED_KEYS = ("image", "input_text", "target_text")


class VLMCollator:
    """Collate raw VLM samples into model-ready tensors.

    Each sample in the batch must be a dict with:
        - ``image``:       ``PIL.Image.Image`` or a path string
        - ``input_text``:  source / prompt text
        - ``target_text``: supervision / label text

    Returns a dict with ``pixel_values``, ``input_ids``,
    ``attention_mask``, and ``labels`` tensors, all contiguous.
    """

    def __init__(self, config: Any) -> None:
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            config.llm_model,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.processor: ViTImageProcessor = ViTImageProcessor.from_pretrained(
            config.vit_model,
        )

        self.max_length: int = config.max_length
        self.image_size: int = getattr(config, "image_size", 224)
        self.debug: bool = getattr(config, "debug", False)

        # Fallback colour for unreadable images (black)
        self._fallback_colour: tuple[int, int, int] = (0, 0, 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_image(self, img: Any) -> Image.Image:
        """Convert *img* to an RGB PIL image, returning a black placeholder on error."""
        try:
            if isinstance(img, Image.Image):
                return img.convert("RGB")
            if isinstance(img, (str, bytes)):
                return Image.open(img).convert("RGB")
            raise TypeError(f"Unsupported image type: {type(img)}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load image (%s); substituting blank image.", exc)
            return Image.new(
                "RGB", (self.image_size, self.image_size), self._fallback_colour
            )

    def _is_valid_sample(self, sample: Any) -> bool:
        """Return True iff *sample* has all required non-None fields."""
        if not isinstance(sample, dict):
            return False
        return all(sample.get(k) is not None for k in _REQUIRED_KEYS)

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _mask_padding_labels(self, label_ids: torch.Tensor) -> torch.Tensor:
        """Replace pad-token positions with _LABEL_IGNORE_INDEX in-place."""
        label_ids[label_ids == self.tokenizer.pad_token_id] = _LABEL_IGNORE_INDEX
        return label_ids

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, batch: list[Any]) -> dict[str, torch.Tensor]:
        """Collate *batch* into tensors.

        Args:
            batch: List of sample dicts from a torch.utils.data.Dataset.

        Returns:
            Dict with keys pixel_values, input_ids, attention_mask, and labels.

        Raises:
            ValueError: If the entire batch is invalid / empty after filtering.
            RuntimeError: If tensor batch dimensions are inconsistent.
        """
        filtered = [s for s in batch if self._is_valid_sample(s)]

        n_dropped = len(batch) - len(filtered)
        if n_dropped:
            logger.warning(
                "Dropped %d/%d malformed sample(s) from batch.", n_dropped, len(batch)
            )

        if not filtered:
            raise ValueError(
                "Batch is empty after filtering invalid samples. "
                "Check that your Dataset returns dicts with "
                "'image', 'input_text', and 'target_text' keys."
            )

        # ---- images ---------------------------------------------------
        images = [self._safe_image(s["image"]) for s in filtered]

        pixel_values: torch.Tensor = self.processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]

        # ---- text tokenization ----------------------------------------
        inputs = [str(s["input_text"]) for s in filtered]
        targets = [str(s["target_text"]) for s in filtered]

        model_inputs = self._tokenize(inputs)
        label_inputs = self._tokenize(targets)

        input_ids: torch.Tensor = model_inputs["input_ids"]
        attention_mask: torch.Tensor = model_inputs["attention_mask"]
        labels: torch.Tensor = self._mask_padding_labels(label_inputs["input_ids"])

        # ---- sanity check ---------------------------------------------
        B = pixel_values.shape[0]
        if not (input_ids.shape[0] == labels.shape[0] == B):
            raise RuntimeError(
                f"Tensor batch-size mismatch — "
                f"pixel_values: {B}, input_ids: {input_ids.shape[0]}, "
                f"labels: {labels.shape[0]}"
            )

        # ---- optional debug logging -----------------------------------
        if self.debug:
            logger.debug(
                "Batch shapes — pixel_values: %s  input_ids: %s  labels: %s",
                tuple(pixel_values.shape),
                tuple(input_ids.shape),
                tuple(labels.shape),
            )

        return {
            "pixel_values": pixel_values.contiguous(),
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "labels": labels.contiguous(),
        }
