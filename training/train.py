import logging
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb
from configs.config import Config
from data.collator import VLMCollator
from data.dataset import VLMDataset
from models.vlm_model import VLMModel
from training.trainer import Trainer

# ----------------------------
# 🔇 CLEAN LOGS
# ----------------------------
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# ----------------------------
# 🔁 REPRODUCIBILITY
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # ---------------- CONFIG ----------------
    config = Config()
    set_seed(config.seed if hasattr(config, "seed") else 42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = config.use_fp16 and torch.cuda.is_available()

    # ---------------- WANDB ----------------
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=vars(config),
    )

    # ---------------- DATA ----------------
    dataset = VLMDataset()
    collator = VLMCollator(config)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers_train,
        pin_memory=pin,
        persistent_workers=config.num_workers_train > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers_val,
        pin_memory=pin,
        persistent_workers=config.num_workers_val > 0,
    )

    # ---------------- MODEL ----------------
    model = VLMModel(config).to(device)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    wandb.watch(model, log="all", log_freq=100)

    # ---------------- OPTIMIZER ----------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # 🔥 FIX: correct total steps (with grad accumulation)
    total_steps = (len(train_loader) * config.epochs) // config.grad_accum_steps

    warmup_steps = int(config.warmup_ratio * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---------------- TRAINER ----------------
    trainer = Trainer(
        model,
        optimizer,
        device=device,
        scheduler=scheduler,
        grad_accum_steps=config.grad_accum_steps,
        use_amp=use_amp,
        use_wandb=True,
    )

    # ---------------- TRAINING ----------------
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    os.makedirs("checkpoints", exist_ok=True)

    global_step = 0

    for epoch in range(config.epochs):
        print(f"\n🚀 Epoch {epoch+1}/{config.epochs}")

        epoch_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc="Training")

        for batch in pbar:
            loss = trainer.train_step(batch)

            if loss is None:
                continue

            epoch_loss += loss
            steps += 1
            global_step += 1

            # progress bar
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        train_loss = epoch_loss / max(steps, 1)

        # ---------------- VALIDATION ----------------
        val_loss = trainer.validate(val_loader)

        print(f"📉 Train Loss: {train_loss:.4f}")
        print(f"📊 Val Loss:   {val_loss:.4f}")

        # ---------------- LOGGING ----------------
        avg_grad = (
            np.mean(trainer.grad_norms[-steps:])
            if trainer.grad_norms and steps > 0
            else 0.0
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "grad_norm": avg_grad,
            },
            step=global_step,
        )

        # ---------------- CHECKPOINT ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            wandb.save("checkpoints/best_model.pt")

            print("✅ Best model saved")

        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{patience})")

            # 🔥 smarter early stopping
            if patience_counter >= patience:
                print("🛑 Early stopping triggered")
                break

    # ---------------- FINAL SAVE ----------------
    torch.save(model.state_dict(), "checkpoints/last_model.pt")
    wandb.save("checkpoints/last_model.pt")

    wandb.finish()

    print("\n🎯 Training Complete")
    print(f"Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
