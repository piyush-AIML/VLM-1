import logging
import random

from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VLMDataset(Dataset):
    def __init__(
        self,
        split="train",
        max_samples=5000,
        seed=42,
        add_ocr_task=True,
        ocr_prob=0.3,  # 🔥 control OCR frequency
    ):
        self.seed = seed
        self.add_ocr_task = add_ocr_task
        self.ocr_prob = ocr_prob

        logger.info("🔄 Loading OCR-VQA dataset...")

        self.ds = load_dataset("howard-hou/OCR-VQA", split=split)

        if max_samples:
            self.ds = self.ds.select(range(min(len(self.ds), max_samples)))

        self.data = self._flatten(self.ds)

        logger.info(f"✅ Final dataset size: {len(self.data)}")

    # --------------------------------------------------
    def __len__(self):
        return len(self.data)

    # --------------------------------------------------
    def __getitem__(self, idx):
        return self.data[idx]

    # --------------------------------------------------
    # 🔥 CORE: FLATTEN + NORMALIZE
    # --------------------------------------------------
    def _flatten(self, dataset):
        random.seed(self.seed)

        processed = []

        for sample in dataset:
            image = sample["image"]

            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            ocr_tokens = sample.get("ocr_tokens", [])

            # ----------------------------
            # ✅ VQA TASK
            # ----------------------------
            for q, a in zip(questions, answers):
                if not q or not a:
                    continue

                processed.append(
                    {
                        "image": image,
                        "input_text": self._format_vqa(q),
                        "target_text": self._clean_text(a),
                    }
                )

            # ----------------------------
            # 🔥 OCR TASK (CONTROLLED)
            # ----------------------------
            if self.add_ocr_task and ocr_tokens and random.random() < self.ocr_prob:
                processed.append(
                    {
                        "image": image,
                        "input_text": "Read all text in the image.",
                        "target_text": self._format_ocr(ocr_tokens),
                    }
                )

        return processed

    # --------------------------------------------------
    # 🧠 PROMPT DESIGN (VERY IMPORTANT)
    # --------------------------------------------------
    def _format_vqa(self, question):
        question = question.strip()

        # normalize question
        if not question.endswith("?"):
            question += "?"

        return f"Question: {question}"

    # --------------------------------------------------
    def _format_ocr(self, tokens):
        # join OCR tokens cleanly
        text = " ".join(tokens)

        # remove duplicates (optional improvement)
        words = text.split()
        words = list(dict.fromkeys(words))  # preserve order

        return " ".join(words)

    # --------------------------------------------------
    def _clean_text(self, text):
        return " ".join(text.strip().split())
