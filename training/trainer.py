import torch


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, batch):
        self.model.train()

        batch = {
            k: v.cuda().float() if k == "pixel_values" else v.cuda()
            for k, v in batch.items()
        }

        if batch is None:
            return 0

        outputs = self.model(**batch)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            print("Skipping NaN batch")
            return 0

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()
