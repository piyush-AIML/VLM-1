# training/train.py

import torch
from torch.utils.data import DataLoader, random_split
from configs.config import Config
from data.dataset import VLMDataset
from data.collator import VLMCollator
from models.vlm_model import VLMModel
from training.trainer import Trainer
import os

# ---------------- CONFIG ----------------
config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATA ----------------
dataset = VLMDataset()
collator = VLMCollator(config)

# split dataset (90% train, 10% val)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collator
)

# ---------------- MODEL ----------------
model = VLMModel(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

trainer = Trainer(model, optimizer, device=device)

# ---------------- TRAINING ----------------
best_val_loss = float("inf")
patience = 3
patience_counter = 0

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(config.epochs):
    print(f"\n🚀 Epoch {epoch+1}/{config.epochs}")

    epoch_loss = 0
    steps = 0

    for batch in train_loader:
        loss = trainer.train_step(batch)

        if loss is not None:
            epoch_loss += loss
            steps += 1

    train_loss = epoch_loss / max(steps, 1)

    # -------- VALIDATION --------
    val_loss = trainer.validate(val_loader)

    print(f"📉 Train Loss: {train_loss:.4f}")
    print(f"📊 Val Loss:   {val_loss:.4f}")

    # -------- GRADIENT INFO --------
    if trainer.grad_norms:
        avg_grad = sum(trainer.grad_norms[-steps:]) / steps
        print(f"📏 Avg Grad Norm: {avg_grad:.4f}")

    # -------- EARLY STOPPING --------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print("✅ Best model saved")

    else:
        patience_counter += 1
        print(f"⏳ No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("🛑 Early stopping triggered")
            break

# ---------------- FINAL SAVE ----------------
torch.save(model.state_dict(), "checkpoints/last_model.pt")

print("\n🎯 Training Complete")
print(f"Best Val Loss: {best_val_loss:.4f}")
