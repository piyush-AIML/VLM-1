import torch 
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer,ViTImageProcessor

class VLMCollator:
    def __init__(self,config):
        self.tokenizer=AutoTokenizer.from_pretrained(config.llm_model)
        self.processor=ViTImageProcessor.from_pretrained(config.vit_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token

    def mask_labels(self,input_ids,tokenizer):
        labels=input_ids.clone

        assistant_token=tokenizer.encode("assistant")[-1]

        mask=torch.ones_like(labels) * -100
        
        for i in range(len(labels)):
            if labels[i] == assistant_token:
                mask[i:] =labels[i:]
        return mask

    def __call__(self,batch):
        images=[b["image"] for b in batch]
        conversations =[b["conversation"] for b in batch]

        pixel_values = self.processor(images=images,return_tensors="pt")[
            "pixel_values"
        ]

        input_ids_list,labels_list=[],[]

        for conv in conversations:
            tokens = self.tokenizer.apply_chat_template(conv,tokenize=True)
            input_ids=torch.tensor(tokens)

            labels = self.mask_labels(input_ids,self.tokenizer)

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        input_ids=pad_sequence(labels_list,batch_first=True,
        padding_value=self.tokenizer.pad_token_id)
        
        labels=pad_sequence(labels_list,batch_first=True,
                            padding_value=-100)

        attention_mask=(input_ids !=self.tokenizer.pad_token_id).long()

        return {
            "pixel_values":pixel_values,
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "labels":labels,
        }

