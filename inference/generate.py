import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor
from models.vlm_model import VLMModel
from configs.config import Config

class VLMInference:
    def __init__(self, checkpoint_path=None):
        self.config = Config()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = VLMModel(self.config).to(self.device)
        self.model.eval()

        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path))

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        self.processor = ViTImageProcessor.from_pretrained(self.config.vit_model)

    def generate(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)

        # ---- TEXT ----
        chat = [
            {"role": "system", "content": "Answer truthfully"},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # ---- IMAGE ----
            vit_out = self.model.vit(pixel_values=pixel_values)
            image_embeds = vit_out.last_hidden_state

            q_tokens = self.model.qformer(image_embeds)
            image_tokens = self.model.proj(q_tokens)

            # ---- TEXT ----
            text_embeds = self.model.llm.get_input_embeddings()(input_ids)

            inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)

            # ---- GENERATE ----
            outputs = self.model.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
