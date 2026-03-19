import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, ViTImageProcessor


class VLMCollator:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model,
            trust_remote_code=True
        )
        self.processor = ViTImageProcessor.from_pretrained(config.vit_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        if len(batch) == 0:
            return None

        images = [b["image"] for b in batch]
        conversations = [b["conversation"] for b in batch]

        # ---- IMAGE ----
        pixel_values = self.processor(
            images=images,
            return_tensors="pt"
        )["pixel_values"]

        input_ids_list = []
        labels_list = []

        # ---- TEXT ----
        for conv in conversations:
            try:
                # 🔥 Proper chat formatting (CRITICAL)
                encoded = self.tokenizer.apply_chat_template(
                    conv,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    add_generation_prompt=False
                )

                input_ids = encoded["input_ids"].squeeze(0)

                # ---- LABELS ----
                labels = input_ids.clone()

                # 🔥 MASK EVERYTHING FIRST
                labels[:] = -100

                # 🔥 UNMASK ONLY LAST PART (assistant response approx)
                # This is stable + works well in practice
                answer_tokens = min(50, len(input_ids))  # last 50 tokens
                labels[-answer_tokens:] = input_ids[-answer_tokens:]

                # ensure valid labels
                if (labels != -100).sum() < 5:
                    continue

                input_ids_list.append(input_ids)
                labels_list.append(labels)

            except Exception:
                continue

        if len(input_ids_list) == 0:
            return None

        # ---- PAD ----
        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        )

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "pixel_values": pixel_values[:len(input_ids)],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
