from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image


class VLMDataset(Dataset):
    def __init__(self,split="train"):
        self.ds=load_dataset("nvidia/Llama-Nemotron-VLM-Dataset-v1",
                             split=split,
                             )
        def __len__(self):
            return len(self.ds)

        def __getItem__(self,idx):
            item=self.ds[idx]

            image=item["image"]
            if not isinstance(image,Image.Image):
                image=Image.open(image).convert("RGB")

            return {
                "image":image,
                "conversation":item["conversation"]
            }
