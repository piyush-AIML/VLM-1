import torch
import torch.nn as nn
from transformers import ViTModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from .qformer import QFormer

class VLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.vit = ViTModel.from_pretrained(config.vit_model)

        self.qformer = QFormer(
            dim=768,
            num_queries=config.num_query_tokens,
            layers=config.qformer_layers
        )

        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model)

        # 🔥 LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        self.llm = get_peft_model(self.llm, lora_config)

        self.proj = nn.Linear(768, self.llm.config.hidden_size)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        vit_out = self.vit(pixel_values=pixel_values)
        image_embeds = vit_out.last_hidden_state

        q_tokens = self.qformer(image_embeds)
        image_tokens = self.proj(q_tokens)

        text_embeds = self.llm.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)

        B = input_ids.size(0)

        image_mask = torch.ones(B, image_tokens.size(1)).to(input_ids.device)
        attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        image_labels = torch.full(
            (B, image_tokens.size(1)),
            -100,
            device=input_ids.device,
        )
        labels = torch.cat([image_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
