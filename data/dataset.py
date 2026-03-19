from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from PIL import Image


class VLMDataset(Dataset):
    def __init__(self, max_samples=1000):
        splits = ["vqa_1", "vqa_2", "captioning_1"]

        datasets_list = [
            load_dataset(
                "nvidia/Llama-Nemotron-VLM-Dataset-v1",
                split=s,
                verification_mode="no_checks"
            )
            for s in splits
        ]

        self.ds = concatenate_datasets(datasets_list)

        # limit dataset size (important for your GPU)
        self.ds = self.ds.select(range(max_samples))

    def __len__(self):
        return len(self.ds)

    def _load_image(self, image):
        """
        Safely load image from dataset
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, str):
            try:
                return Image.open(image).convert("RGB")
            except Exception:
                return None

        return None

    def __getitem__(self, idx):
        # 🔥 NO recursion — use loop
        max_tries = 10

        for i in range(max_tries):
            item = self.ds[(idx + i) % len(self.ds)]

            img = self._load_image(item["image"])

            if img is not None:
                return {
                    "image": img,
                    "conversation": item["conversations"]
                }

        # 🚨 fallback (rare case)
        # return a dummy safe sample
        return {
            "image": Image.new("RGB", (224, 224), color=(0, 0, 0)),
            "conversation": [
                {"role": "user", "content": "Describe the image"},
                {"role": "assistant", "content": "A blank image"}
            ]
        }
