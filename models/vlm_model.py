import logging

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, ViTModel

from .qformer import QFormer

logger = logging.getLogger(__name__)


class VLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # =========================================================
        # 🖼️ VISION ENCODER (FROZEN)
        # =========================================================
        self.vit = ViTModel.from_pretrained(config.vit_model)
        self.vision_dim = self.vit.config.hidden_size

        for p in self.vit.parameters():
            p.requires_grad = False

        # =========================================================
        # 🧠 Q-FORMER
        # =========================================================
        self.qformer = QFormer(
            dim=self.vision_dim,
            num_queries=config.num_query_tokens,
            layers=config.qformer_layers,
        )

        # =========================================================
        # 🧠 LLM (LoRA)
        # =========================================================
        use_fp16 = config.use_fp16 and torch.cuda.is_available()

        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
        )

        if self.llm.config.pad_token_id is None:
            self.llm.config.pad_token_id = self.llm.config.eos_token_id or 0

        # 🔥 LoRA config
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_targets or ["q_proj", "v_proj"],
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.llm = get_peft_model(self.llm, lora_config)

        if config.gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()

        self.llm_dim = self.llm.config.hidden_size

        # =========================================================
        # 🔗 PROJECTION (VISION → LLM SPACE)
        # =========================================================
        self.proj = nn.Linear(self.vision_dim, self.llm_dim)
        self.image_norm = nn.LayerNorm(self.llm_dim)

        # 🔥 learnable scaling
        self.image_scale = nn.Parameter(torch.ones(1) * 0.1)

        # 🔥 fusion norm (critical)
        self.fusion_norm = nn.LayerNorm(self.llm_dim)

        self._log_trainable_params()

    # =========================================================
    # 📊 PARAM STATS
    # =========================================================
    def _log_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(
            f"Trainable params: {trainable:,}/{total:,} "
            f"({100*trainable/total:.2f}%)"
        )

    # =========================================================
    # 🚀 FORWARD (TRAINING)
    # =========================================================
    def forward(self, pixel_values, input_ids, attention_mask, labels):
        device = input_ids.device
        B = input_ids.size(0)

        # ----------------------------
        # 🖼️ Vision → embeddings
        # ----------------------------
        with torch.no_grad():
            image_embeds = self.vit(
                pixel_values=pixel_values
            ).last_hidden_state  # (B, N, D)

        # ----------------------------
        # 🧠 Q-Former
        # ----------------------------
        q_tokens = self.qformer(image_embeds)  # (B, Q, D)

        # ----------------------------
        # 🔗 Project to LLM space
        # ----------------------------
        image_tokens = self.proj(q_tokens)
        image_tokens = self.image_norm(image_tokens)
        image_tokens = image_tokens * self.image_scale

        # ----------------------------
        # 🔤 Prepare text (teacher forcing)
        # ----------------------------
        pad_id = self.llm.config.pad_token_id

        target_ids = labels.clone()
        target_ids[target_ids == -100] = pad_id

        combined_ids = torch.cat([input_ids, target_ids], dim=1)

        # ----------------------------
        # 🔤 Correct attention mask
        # ----------------------------
        text_mask = (combined_ids != pad_id).long()

        # ----------------------------
        # 🎯 Labels (mask input part)
        # ----------------------------
        new_labels = combined_ids.clone()
        input_len = input_ids.size(1)
        new_labels[:, :input_len] = -100

        # ----------------------------
        # 🧠 Get embeddings
        # ----------------------------
        text_embeds = self.llm.get_input_embeddings()(combined_ids)
        image_tokens = image_tokens.to(text_embeds.dtype)

        # ----------------------------
        # 🔗 Fusion (image prefix)
        # ----------------------------
        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)
        inputs_embeds = self.fusion_norm(inputs_embeds)

        # ----------------------------
        # 🎯 Final attention mask
        # ----------------------------
        image_mask = torch.ones(
            B,
            image_tokens.size(1),
            device=device,
            dtype=text_mask.dtype,
        )

        attention_mask = torch.cat([image_mask, text_mask], dim=1)

        # ----------------------------
        # 🎯 Final labels
        # ----------------------------
        image_labels = torch.full(
            (B, image_tokens.size(1)),
            -100,
            device=device,
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

    # =========================================================
    # 🤖 GENERATION
    # =========================================================
    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        max_new_tokens=50,
        **gen_kwargs,
    ):
        device = input_ids.device

        # Vision
        image_embeds = self.vit(pixel_values=pixel_values).last_hidden_state
        q_tokens = self.qformer(image_embeds)

        image_tokens = self.proj(q_tokens)
        image_tokens = self.image_norm(image_tokens)
        image_tokens = image_tokens * self.image_scale

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_tokens = image_tokens.to(text_embeds.dtype)

        inputs_embeds = torch.cat([image_tokens, text_embeds], dim=1)
        inputs_embeds = self.fusion_norm(inputs_embeds)

        image_mask = torch.ones(
            input_ids.size(0),
            image_tokens.size(1),
            device=device,
            dtype=attention_mask.dtype,
        )

        attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
