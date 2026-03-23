import logging

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, ViTModel

from .qformer import QFormer

logging.basicConfig(level=logging.INFO)


class VLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # ----------------------------
        # 🖼️ Vision Encoder
        # ----------------------------
        self.vit = ViTModel.from_pretrained(config.vit_model)
        self.vision_dim = self.vit.config.hidden_size

        for p in self.vit.parameters():
            p.requires_grad = False

        # ----------------------------
        # 🧠 Q-Former
        # ----------------------------
        self.qformer = QFormer(
            dim=self.vision_dim,
            num_queries=config.num_query_tokens,
            layers=config.qformer_layers,
        )

        nn.init.normal_(self.qformer.query_tokens, std=0.02)

        # ----------------------------
        # 🧠 LLM (fp16 only on CUDA; CPU + fp16 weights breaks many ops / dtypes)
        # ----------------------------
        _llm_half = bool(config.use_fp16 and torch.cuda.is_available())
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            torch_dtype=torch.float16 if _llm_half else torch.float32,
        )

        # 🔥 FIX: pad_token_id safety
        if self.llm.config.pad_token_id is None:
            self.llm.config.pad_token_id = 0

        # ----------------------------
        # 🔥 LoRA
        # ----------------------------
        target_modules = config.lora_targets or ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.llm = get_peft_model(self.llm, lora_config)

        if config.gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

        self.llm_dim = self.llm.config.hidden_size

        # ----------------------------
        # 🔗 Projection
        # ----------------------------
        self.proj = nn.Linear(self.vision_dim, self.llm_dim)
        self.image_norm = nn.LayerNorm(self.llm_dim)
        self.image_scale = nn.Parameter(torch.tensor(0.1))

        self._log_trainable_params()

    # ----------------------------
    # 📊 Trainable Params
    # ----------------------------
    def _log_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logging.info(
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    # ----------------------------
    # 🚀 Forward (FINAL FIXED)
    # ----------------------------
    def forward(self, pixel_values, input_ids, attention_mask, labels):
        B = input_ids.size(0)

        # ----------------------------
        # 🖼️ Vision
        # ----------------------------
        with torch.no_grad():
            image_embeds = self.vit(pixel_values=pixel_values).last_hidden_state

        # ----------------------------
        # 🧠 Q-Former
        # ----------------------------
        q_tokens = self.qformer(image_embeds)

        # ----------------------------
        # 🔗 Projection
        # ----------------------------
        image_tokens = self.proj(q_tokens)
        image_tokens = self.image_norm(image_tokens)
        image_tokens = image_tokens * self.image_scale

        # ----------------------------
        # 🔥 FIX 1: CLEAN TARGET IDS
        # ----------------------------
        pad_id = self.llm.config.pad_token_id
        target_ids = labels.clone()

        target_ids[target_ids == -100] = pad_id  # FIXED

        # ----------------------------
        # 🔥 FIX 2: COMBINE TEXT
        # ----------------------------
        combined_ids = torch.cat([input_ids, target_ids], dim=1)

        # ----------------------------
        # 🔥 FIX 3: LABEL MASKING
        # ----------------------------
        new_labels = combined_ids.clone()
        input_len = input_ids.size(1)
        new_labels[:, :input_len] = -100

        # ----------------------------
        # 🧠 Text embeddings
        # ----------------------------
        text_embeds = self.llm.get_input_embeddings()(combined_ids)
        image_tokens = image_tokens.to(text_embeds.dtype)

        # ----------------------------
        # 🔗 Combine image + text
        # ----------------------------
        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)

        # ----------------------------
        # 🎯 Attention mask
        # ----------------------------
        text_mask = torch.ones_like(combined_ids)

        image_mask = torch.ones(
            B,
            image_tokens.size(1),
            device=input_ids.device,
            dtype=text_mask.dtype,
        )

        attention_mask = torch.cat([image_mask, text_mask], dim=1)

        # ----------------------------
        # 🎯 Labels
        # ----------------------------
        image_labels = torch.full(
            (B, image_tokens.size(1)),
            -100,
            device=input_ids.device,
        )

        labels = torch.cat([image_labels, new_labels], dim=1)

        # ----------------------------
        # 🚀 Forward
        # ----------------------------
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    # ----------------------------
    # 🤖 Generation
    # ----------------------------
    @torch.no_grad()
    def generate(
        self, pixel_values, input_ids, attention_mask, max_new_tokens=50, **gen_kwargs
    ):
        image_embeds = self.vit(pixel_values=pixel_values).last_hidden_state

        q_tokens = self.qformer(image_embeds)

        image_tokens = self.proj(q_tokens)
        image_tokens = self.image_norm(image_tokens)
        image_tokens = image_tokens * self.image_scale

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_tokens = image_tokens.to(text_embeds.dtype)

        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)

        image_mask = torch.ones(
            input_ids.size(0),
            image_tokens.size(1),
            device=input_ids.device,
            dtype=attention_mask.dtype,
        )

        attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        gen_kwargs = {"max_new_tokens": max_new_tokens, **gen_kwargs}
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
