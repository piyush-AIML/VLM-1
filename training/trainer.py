# training/trainer.py

import torch


class Trainer:
    def __init__(self, model, optimizer, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # tracking
        self.train_losses = []
        self.val_losses = []
        self.grad_norms = []

    def _move_batch(self, batch):
        return {
            k: v.to(self.device).float() if k == "pixel_values" else v.to(self.device)
            for k, v in batch.items()
        }

    def train_step(self, batch):
        self.model.train()

        batch = self._move_batch(batch)

        outputs = self.model(**batch)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Skipping NaN batch")
            return None

        loss.backward()

        # gradient norm (for monitoring stability)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.grad_norms.append(grad_norm.item())

        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_value = loss.item()
        self.train_losses.append(loss_value)

        return loss_value

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()

        batch = self._move_batch(batch)

        outputs = self.model(**batch)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            return None

        return loss.item()

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()

        total_loss = 0
        count = 0

        for batch in dataloader:
            loss = self.eval_step(batch)
            if loss is not None:
                total_loss += loss
                count += 1

        avg_loss = total_loss / max(count, 1)
        self.val_losses.append(avg_loss)

        return avg_loss
