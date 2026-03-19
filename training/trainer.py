import torch
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()

    def train_step(self, batch):
        self.model.train()

        batch = {k: v.cuda() for k, v in batch.items()}

        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return loss.item()
