import logging
import time
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        device: str = "cuda",
        scheduler=None,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        use_wandb: bool = True,
        log_interval: int = 10,
    ):
        # ----------------------------
        # 🔒 Device setup (FORCE GPU)
        # ----------------------------
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

        self._is_cuda = self.device.type == "cuda"
        self._amp_device_type = "cuda" if self._is_cuda else "cpu"

        # ----------------------------
        # ⚡ AMP
        # ----------------------------
        self.scaler = GradScaler(enabled=use_amp and self._is_cuda)

        # ----------------------------
        # 📊 Tracking
        # ----------------------------
        self.train_losses = []
        self.val_losses = []
        self.grad_norms = []

        self.step_count = 0
        self.opt_step_count = 0

        self.log_interval = log_interval
        self._last_log_time = time.time()

        # ----------------------------
        # 📡 WandB
        # ----------------------------
        self.use_wandb = use_wandb and _WANDB_AVAILABLE

        if use_wandb and not _WANDB_AVAILABLE:
            logger.warning("wandb not installed → logging disabled")

        logger.info(f"Using device: {self.device}")
        logger.info(f"WandB: {'ON' if self.use_wandb else 'OFF'}")

        # optional debug (disable in production)
        # torch.autograd.set_detect_anomaly(True)

    # --------------------------------------------------
    # 📦 Move batch to GPU
    # --------------------------------------------------
    def _move_batch(self, batch: dict[str, Any]):
        return {
            k: (
                v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

    def _current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _is_accumulation_step(self):
        return (self.step_count + 1) % self.grad_accum_steps == 0

    def _log(self, data):
        if self.use_wandb:
            try:
                wandb.log(data)
            except Exception as e:
                logger.warning(f"WandB logging failed: {e}")

    # --------------------------------------------------
    # 🚀 TRAIN STEP
    # --------------------------------------------------
    def train_step(self, batch):
        self.model.train()
        batch = self._move_batch(batch)

        start_time = time.time()

        # ----------------------------
        # 🔥 Forward (AMP)
        # ----------------------------
        with autocast(
            device_type=self._amp_device_type,
            enabled=self.use_amp,
        ):
            outputs = self.model(**batch)

            if outputs.loss is None:
                raise ValueError("❌ Model returned loss=None")

            loss = outputs.loss / self.grad_accum_steps

        # ----------------------------
        # 🔒 NaN/Inf guard
        # ----------------------------
        if not torch.isfinite(loss):
            logger.warning(f"Skipping invalid loss at step {self.step_count}")
            self.optimizer.zero_grad(set_to_none=True)
            return None

        # ----------------------------
        # 🔙 Backward
        # ----------------------------
        self.scaler.scale(loss).backward()

        grad_norm = None

        # ----------------------------
        # ⚙️ Optimizer step
        # ----------------------------
        if self._is_accumulation_step():

            self.scaler.unscale_(self.optimizer)

            if self.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                grad_norm = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
                self.grad_norms.append(grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler:
                self.scheduler.step()

            self.opt_step_count += 1

        self.step_count += 1

        # ----------------------------
        # 📊 Metrics
        # ----------------------------
        loss_value = loss.item() * self.grad_accum_steps
        self.train_losses.append(loss_value)

        step_time = time.time() - start_time
        lr = self._current_lr()

        # ----------------------------
        # 📡 Logging
        # ----------------------------
        if self.step_count % self.log_interval == 0:
            log_data = {
                "train/loss": loss_value,
                "train/lr": lr,
                "train/grad_norm": grad_norm if grad_norm else 0,
                "train/step": self.step_count,
                "train/step_time": step_time,
            }
            self._log(log_data)

        # ----------------------------
        # 🔥 GPU memory cleanup (important for 6GB GPUs)
        # ----------------------------
        if self._is_cuda and self.step_count % 100 == 0:
            torch.cuda.empty_cache()

        return loss_value

    # --------------------------------------------------
    # 🔍 EVAL STEP
    # --------------------------------------------------
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

        if loss is None or not torch.isfinite(loss):
            return None

        return loss.item()

    # --------------------------------------------------
    # 📊 VALIDATION LOOP
    # --------------------------------------------------
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

        if count == 0:
            raise RuntimeError("❌ No valid validation batches")

        avg_loss = total_loss / count
        self.val_losses.append(avg_loss)

        logger.info(f"Validation Loss: {avg_loss:.4f}")

        self._log(
            {
                "val/loss": avg_loss,
                "train/step": self.step_count,
            }
        )

        return avg_loss
