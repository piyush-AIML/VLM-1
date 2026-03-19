from torch.utils.data import DataLoader
import torch
from configs.config import Config
from data.dataset import VLMDataset
from data.collator import VLMCollator
from models.vlm_model import VLMModel
from training.trainer import Trainer

config = Config()

dataset = VLMDataset()
collator = VLMCollator(config)

loader = DataLoader(dataset, batch_size=config.batch_size,
                    shuffle=True, collate_fn=collator)

model = VLMModel(config).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

trainer = Trainer(model, optimizer)

for epoch in range(config.epochs):
    for batch in loader:
        loss = trainer.train_step(batch)
        print("Loss:", loss)


torch.save(model.state_dict(),"checkpoints/model.pt")
