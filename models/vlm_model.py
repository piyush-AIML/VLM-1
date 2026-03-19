import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, AutoModelForCausalLM
# 🔥 temporarily REMOVE LoRA
# from peft import LoraConfig, get_peft_model
from .qformer import QFormer


class VLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # ---- Vision ----
        self.vit = ViTModel.from_pretrained(config.vit_model,dtype=torch.float32)

        # 🔥 freeze ViT (VERY IMPORTANT)
        for p in self.vit.parameters():
            p.requires_grad = False

        # ---- Q-Former ----
        self.qformer = QFormer(
            dim=768,
            num_queries=config.num_query_tokens,
            layers=config.qformer_layers
        )

        # ---- LLM ----
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model,dtype=torch.float32)

        # 🔥 freeze LLM (stability phase)
        for p in self.llm.parameters():
            p.requires_grad = False

        # ---- Projection ----
        self.proj = nn.Linear(768, self.llm.config.hidden_size)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        # ---- Vision ----
        with torch.no_grad():  # 🔥 stability
            vit_out = self.vit(pixel_values=pixel_values)
            image_embeds = vit_out.last_hidden_state

        # ---- Q-Former ----
        q_tokens = self.qformer(image_embeds)

        # ---- Projection ----
        image_tokens = self.proj(q_tokens)

        # 🔥 CRITICAL FIXES (NaN prevention)
        image_tokens = F.normalize(image_tokens, dim=-1)
        image_tokens = image_tokens * 0.1

        # ---- Text Embedding ----
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # ---- Combine ----
        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)

        # 🔥 clamp to prevent explosion
        inputs_embeds = torch.clamp(inputs_embeds, -10, 10)

        # ---- Masks ----
        B = input_ids.size(0)

        image_mask = torch.ones(
            B, image_tokens.size(1),
            device=input_ids.device
        )

        attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        # ---- Labels ----
        image_labels = torch.full(
            (B, image_tokens.size(1)),
            -100,
            device=input_ids.device,
        )

        labels = torch.cat([image_labels, labels], dim=1)

        # ---- Forward ----
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs
