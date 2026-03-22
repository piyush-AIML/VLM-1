import logging
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = _PROJECT_ROOT / "data_cache"


class VLMDataset(Dataset):
    def __init__(self, max_samples=20000, seed=42, cache_dir=None):
        splits = [
            "vqa_1",
            "vqa_2",
            "captioning_1",
            "ocr_1",
            "ocr_2",
            "ocr_3",
            "ocr_4",
        ]

        self._cache = str(cache_dir or _DEFAULT_CACHE)

        logging.info("🔄 Loading datasets...")

        ds_list = [
            load_dataset(
                "nvidia/Llama-Nemotron-VLM-Dataset-v1",
                split=s,
                verification_mode="no_checks",
                cache_dir=self._cache,
            )
            for s in splits
        ]

        ds = concatenate_datasets(ds_list)
        ds = ds.shuffle(seed=seed)

        if max_samples:
            ds = ds.select(range(max_samples))

        logging.info(f"Raw dataset size: {len(ds)}")

        self.samples = self._build_samples(ds)

        logging.info(f"✅ Final usable samples: {len(self.samples)}")

    # ----------------------------
    # 🔥 PARSER (NO IMAGE LOADING)
    # ----------------------------
    def _extract_sample(self, item):

        image = item.get("image", None)  # keep raw

        # ----------------------------
        # VQA
        # ----------------------------
        if "conversations" in item:
            user, assistant = [], []

            for turn in item["conversations"]:
                role = turn.get("role") or turn.get("from")
                text = turn.get("content") or turn.get("value")

                if not text:
                    continue

                text = text.replace("<image>", "").strip()

                if role in ["user", "human"]:
                    user.append(text)
                elif role in ["assistant", "gpt"]:
                    assistant.append(text)

            if user and assistant:
                return image, " ".join(user), " ".join(assistant)

        # ----------------------------
        # ALT VQA
        # ----------------------------
        if "question" in item and "answer" in item:
            return image, item["question"], item["answer"]

        # ----------------------------
        # Caption
        # ----------------------------
        if "caption" in item:
            return image, "Describe the image.", item["caption"]

        # ----------------------------
        # OCR
        # ----------------------------
        for key in ["text", "ocr", "label"]:
            if key in item and isinstance(item[key], str):
                return image, "Read the text in the image.", item[key]

        return None, None, None

    # ----------------------------
    # 🔧 BUILD DATASET
    # ----------------------------
    def _build_samples(self, ds):
        samples = []
        valid, invalid = 0, 0

        for idx, item in enumerate(ds):

            if idx < 3:
                logging.debug("sample item keys=%s", list(item.keys()) if hasattr(item, "keys") else type(item))

            img, inp, tgt = self._extract_sample(item)

            if inp is None or tgt is None:
                invalid += 1
                continue

            samples.append(
                {
                    "image": img,  # may be string
                    "input_text": inp,
                    "target_text": tgt,
                }
            )

            valid += 1

        logging.info(f"Valid: {valid}, Invalid: {invalid}")

        if len(samples) == 0:
            raise RuntimeError("No valid samples!")

        return samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
