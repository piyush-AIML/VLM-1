from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor


class VLMCollator:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model)

        # ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.processor = ViTImageProcessor.from_pretrained(config.vit_model)
        self.max_length = config.max_length

    # ----------------------------
    # 🖼️ SAFE IMAGE HANDLER
    # ----------------------------
    def _safe_image(self, img):
        try:
            # already PIL image
            if hasattr(img, "convert"):
                return img.convert("RGB")

            # ❗ string or invalid → fallback
            return Image.new("RGB", (224, 224), color=(0, 0, 0))

        except Exception:
            return Image.new("RGB", (224, 224), color=(0, 0, 0))

    # ----------------------------
    # 📦 MAIN COLLATE
    # ----------------------------
    def __call__(self, batch):

        # ----------------------------
        # 🔒 FILTER BAD TEXT SAMPLES
        # ----------------------------
        batch = [
            b
            for b in batch
            if b is not None
            and "input_text" in b
            and "target_text" in b
            and b["input_text"] is not None
            and b["target_text"] is not None
        ]

        if len(batch) == 0:
            raise ValueError("❌ Empty batch")

        # ----------------------------
        # 🖼️ IMAGES (SAFE)
        # ----------------------------
        images = [self._safe_image(b["image"]) for b in batch]

        # ----------------------------
        # 🔤 TEXT
        # ----------------------------
        inputs = [str(b["input_text"]) for b in batch]
        targets = [str(b["target_text"]) for b in batch]

        # ----------------------------
        # 🔤 TOKENIZE INPUT
        # ----------------------------
        model_inputs = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # ----------------------------
        # 🔤 TOKENIZE TARGET
        # ----------------------------
        labels = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]

        # ignore padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        # ----------------------------
        # 🖼️ PROCESS IMAGES
        # ----------------------------
        pixel_values = self.processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]

        # ----------------------------
        # 🔒 FINAL SAFETY CHECK
        # ----------------------------
        B = pixel_values.shape[0]

        if model_inputs["input_ids"].shape[0] != B:
            raise RuntimeError("Batch mismatch (text vs image)")

        # ----------------------------
        # 📦 RETURN
        # ----------------------------
        return {
            "pixel_values": pixel_values.contiguous(),
            "input_ids": model_inputs["input_ids"].contiguous(),
            "attention_mask": model_inputs["attention_mask"].contiguous(),
            "labels": labels.contiguous(),
        }
