# 🧠 VLM-1: Vision-Language Model with Q-Former Architecture

A modular, scalable implementation of a **Vision-Language Model (VLM)** integrating a pretrained Vision Transformer with a language model via a **Q-Former** architecture.

This project is designed for **multi-task vision-language learning**, including:

- Visual Question Answering (VQA)
- Image Captioning
- Optical Character Recognition (OCR)

---

## 📌 Features

- 🔗 **Q-Former-based architecture** for efficient vision-language alignment
- 🧩 Modular pipeline (Dataset → Collator → Model → Trainer)
- 📊 Integrated **Weights & Biases (wandb)** tracking
- ⚡ Support for large-scale datasets (Hugging Face Arrow format)
- 🔄 Multi-task training (VQA, OCR, Captioning)
- 🧪 Debug utilities for pipeline inspection

---

## 🏗️ Project Structure

```
VLM-1/
│
├── configs/                # Configuration files
├── data/                   # Dataset + Collator
│   ├── dataset.py
│   └── collator.py
│
├── models/                 # Model architecture
│   ├── qformer.py
│   └── vlm_model.py
│
├── training/               # Training pipeline
│   ├── train.py
│   └── trainer.py
│
├── inference/              # Inference scripts
│   └── generate.py
│
├── evaluation/             # Metrics and evaluation
│   └── metrics.py
│
├── debug/                  # Debug utilities
├── checkpoints/            # Saved models
├── data_cache/             # Hugging Face dataset cache
└── utils/                  # Misc utilities
```

---

## 🧠 Model Architecture

The model follows a **BLIP-2 style pipeline**:

```
Image → Vision Encoder (ViT)
        ↓
     Q-Former
        ↓
   Language Model (LLM)
        ↓
   Text Output
```

### Components

- **Vision Encoder**
  - Extracts image features (patch embeddings)

- **Q-Former**
  - Learns query tokens to bridge vision and language
  - Reduces dimensional mismatch
  - Enables efficient cross-modal attention

- **Language Model**
  - Generates output (captions, answers, OCR text)

---

## 📊 Supported Tasks

| Task       | Description                   |
| ---------- | ----------------------------- |
| VQA        | Answer questions about images |
| Captioning | Generate image descriptions   |
| OCR        | Extract text from images      |

---

## 📦 Dataset

This project supports multiple datasets via Hugging Face:

- `nvidia/llama-nemotron-vlm-dataset-v1`
- `howard-hou/OCR-VQA`

Datasets are automatically:

- Downloaded
- Cached as `.arrow` files
- Loaded efficiently using Hugging Face `datasets`

---

## ⚙️ Installation

```bash
git clone https://github.com/piyush-AIML/VLM-1.git
cd VLM-1

pip install -r requirements.txt
```

---

## 🔑 Hugging Face Authentication (Recommended)

```bash
hf auth login
```

Prevents:

- Rate limits
- Slow downloads
- Dataset failures

---

## 🚀 Training

```bash
python -m training.train
```

### Training Features:

- Gradient tracking
- Learning rate scheduling
- WandB logging
- Checkpoint saving

---

## 📈 Monitoring

Training logs are tracked using **Weights & Biases**:

```bash
wandb login
```

Track:

- Loss
- Gradient norm
- Learning rate
- Step time

---

## 🧪 Inference

```bash
python -m inference.generate
```

Generates:

- Image captions
- Answers
- OCR outputs

---

## 📊 Evaluation

```bash
python -m evaluation.metrics
```

Supports:

- Accuracy
- Text similarity metrics

---

## ⚡ Performance Notes

- Use SSD for faster data loading
- Enable multiple workers in DataLoader
- Use gradient clipping for stability

---

## ⚠️ Known Issues

- Dataset downloads may fail if interrupted
- Hugging Face cache corruption possible
- High gradient spikes during early training

---

## 🛠️ Future Improvements

- [ ] Add distributed training (DDP)
- [ ] Mixed precision training (FP16)
- [ ] Better dataset balancing strategy
- [ ] Model checkpoint versioning
- [ ] Deployment-ready inference API

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch
3. Submit a PR

---

## 📜 License

MIT License

---

## 👤 Author

**Piyush Ghosal**
B.Tech CSE (AI & ML)
MAKAUT University

---

## ⭐ Acknowledgements

Inspired by:

- BLIP / BLIP-2 architectures
- Hugging Face ecosystem
- Vision-Language research advancements

---
