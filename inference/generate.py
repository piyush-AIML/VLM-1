import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor

from configs.config import Config
from models.vlm_model import VLMModel


class VLMInference:
    def __init__(self, checkpoint_path=None):
        self.config = Config()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = VLMModel(self.config).to(self.device)
        self.model.eval()

        if checkpoint_path:
            try:
                state = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        self.processor = ViTImageProcessor.from_pretrained(self.config.vit_model)

    def generate(self, image_path, prompt):
        with Image.open(image_path) as im:
            image = im.convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)

        chat = [
            {"role": "system", "content": "Answer truthfully"},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt"
        ).to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                input_ids,
                attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
