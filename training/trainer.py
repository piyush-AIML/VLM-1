import torch
from torch.amp import GradScaler, autocast


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        device="cuda",
        scheduler=None,
        grad_accum_steps=1,
        max_grad_norm=1.0,
        use_amp=True,  # 🔥 NEW (control AMP)
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self._cuda = str(device).startswith("cuda")
        self._amp_device_type = "cuda" if self._cuda else "cpu"

        # ✅ AMP scaler only on CUDA; CPU keeps fp32 and no loss scaling
        self.scaler = GradScaler(enabled=use_amp and self._cuda)

        # tracking
        self.train_losses = []
        self.val_losses = []
        self.grad_norms = []

        self.step_count = 0

    # ----------------------------
    # 📦 Move batch to device
    # ----------------------------
    def _move_batch(self, batch):
        nb = self._cuda
        return {
            k: v.to(self.device, non_blocking=nb) if hasattr(v, "to") else v
            for k, v in batch.items()
        }

    # ----------------------------
    # 🚀 Train Step
    # ----------------------------
    def train_step(self, batch):
        self.model.train()

        batch = self._move_batch(batch)

        with autocast(
            device_type=self._amp_device_type,
            enabled=self.use_amp,
        ):
            outputs = self.model(**batch)
            loss = outputs.loss
            loss = loss / self.grad_accum_steps

        # 🔥 NaN/Inf guard
        if not torch.isfinite(loss):
            print("⚠️ Skipping invalid loss batch")
            self.optimizer.zero_grad(set_to_none=True)
            return None

        # ----------------------------
        # backward
        # ----------------------------
        self.scaler.scale(loss).backward()

        # ----------------------------
        # optimizer step (grad accumulation)
        # ----------------------------
        if (self.step_count + 1) % self.grad_accum_steps == 0:

            # unscale before clipping
            self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

            self.grad_norms.append(
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )

            # step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

        self.step_count += 1

        loss_value = loss.item() * self.grad_accum_steps
        self.train_losses.append(loss_value)

        return loss_value

    # ----------------------------
    # 🔍 Eval Step
    # ----------------------------
    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()

        batch = self._move_batch(batch)

        with autocast(
            device_type=self._amp_device_type,
            enabled=self.use_amp,
        ):
            outputs = self.model(**batch)
            loss = outputs.loss

        if not torch.isfinite(loss):
            return None

        return loss.item()

    # ----------------------------
    # 📊 Validation Loop
    # ----------------------------
    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()

        total_loss = 0.0
        count = 0

        for batch in dataloader:
            loss = self.eval_step(batch)

            if loss is not None:
                total_loss += loss
                count += 1

        avg_loss = total_loss / max(count, 1)
        self.val_losses.append(avg_loss)

        return avg_loss
