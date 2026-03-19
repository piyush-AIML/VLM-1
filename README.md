I am building a Vision-Language Model (VLM) from scratch using a modular deep learning pipeline. The goal is to create a mini LLaVA-style architecture that takes an image + text prompt and generates a textual response.

## 🧠 PROJECT OVERVIEW

This project implements a multimodal system combining:

- Vision encoder (ViT)
- Query transformer (Q-Former)
- Large Language Model (Qwen)
- Projection layer for modality alignment

The system follows a standard VLM architecture where:

- Image → converted into tokens via ViT
- Tokens → compressed using Q-Former
- Projected into LLM embedding space
- Concatenated with text tokens
- LLM generates output autoregressively

This aligns with modern VLM design where a vision encoder + LLM are fused via projection or cross-attention. ([Cameron R. Wolfe][1])

---

## 🏗️ CURRENT ARCHITECTURE

### 1. Vision Encoder

- Model: ViT (google/vit-base-patch16-224)
- Frozen during training
- Outputs patch embeddings

### 2. Q-Former

- Transformer encoder with learnable query tokens
- Extracts relevant visual features
- Acts as bottleneck between vision and language

### 3. Projection Layer

- Linear layer mapping 768 → LLM hidden size
- Followed by:
  - normalization
  - scaling (×0.1)
  - clamping (for stability)

### 4. Language Model

- Model: Qwen2 (0.5B)
- Initially frozen → later LoRA-enabled
- Takes combined embeddings

### 5. Fusion Strategy

- Early fusion via concatenation:
  image_tokens + text_embeddings → LLM

---

## 📦 DATA PIPELINE

### Dataset

- Source: nvidia/Llama-Nemotron-VLM-Dataset-v1
- Using splits like:
  - vqa_1
  - captioning_1

- Subsampled (1000–20000 samples for experiments)

### Dataset Handling

- Skips corrupted/missing images
- Returns:
  {
  image: PIL.Image,
  conversation: structured chat
  }

---

## 🧠 COLLATOR DESIGN (CRITICAL)

Key features:

- Uses tokenizer.apply_chat_template() (NOT manual string formatting)
- Converts conversations into proper LLM chat format
- Generates:
  - input_ids
  - attention_mask
  - labels

### Label Strategy

- All tokens masked (-100)
- Only last ~50 tokens (assistant response) unmasked

Goal:

- Train ONLY on assistant outputs
- Avoid label collapse / NaN

---

## ⚙️ TRAINING PIPELINE

### Trainer

- Pure FP32 (NO AMP)
- Gradient clipping (1.0)
- NaN detection + skip
- AdamW optimizer

### Stability Fixes Applied

- Removed AMP (causing NaN)
- Fixed dtype mismatch (float32 everywhere)
- Normalized image embeddings
- Scaled image tokens
- Clamped embeddings
- Fixed label masking

---

## 📉 TRAINING BEHAVIOR OBSERVED

### Phase 1 (Correct)

Loss:
5 → 3 → 2 → 1

Meaning:
✔ model learning

---

### Phase 2 (Overfitting)

Loss:
1 → 0.3 → 0.00005

Meaning:
❌ model memorizing
❌ weak training signal
❌ too few tokens contributing to loss

---

## 🚨 CURRENT STATE

✔ Training is stable
✔ Loss decreases correctly
✔ No NaN / crashes
❌ Model overfits quickly
❌ Loss becomes too small
❌ Not generalizing yet

---

## 🎯 CURRENT GOALS

Next steps:

1. Enable LoRA for efficient fine-tuning
2. Improve label masking (true assistant-only training)
3. Increase dataset size/diversity
4. Add validation + early stopping
5. Build inference pipeline
6. Improve multimodal alignment

---

## 🧠 KEY LEARNINGS FROM THIS PROJECT

- VLM = vision encoder + LLM + alignment module ([IBM][2])
- Most failures come from:
  - bad masking
  - dtype mismatch
  - unstable fusion

- Low loss ≠ good model
- Data quality > architecture complexity
- Training stability is harder than model design

---

## 🧩 EXPECTATIONS FOR NEXT SESSION

When continuing this project:

- Assume pipeline is working and stable
- Focus on improving model quality, not debugging
- Prioritize:
  - LoRA integration
  - better supervision signals
  - inference quality

---

## 🧠 ROLE OF ASSISTANT

When helping:

- Understand full architecture (ViT + Q-Former + LLM)
- Suggest improvements at system level, not just code fixes
- Avoid naive solutions (string tokenization, random masking)
- Focus on real VLM practices (like LLaVA-style training)

---

## 🚀 FINAL CONTEXT

This is not a beginner project.

This is a:
→ full custom Vision-Language Model training system
→ built from scratch
→ currently transitioning from “working” → “useful”

I want to move toward:
→ real multimodal reasoning
→ strong VQA / captioning capability
→ efficient fine-tuning (LoRA)

---

Continue from this state without re-explaining basics.

[1]: https://cameronrwolfe.substack.com/p/vision-llms?utm_source=chatgpt.com "Vision Large Language Models (vLLMs)"
[2]: https://www.ibm.com/think/topics/vision-language-models?utm_source=chatgpt.com "What Are Vision Language Models (VLMs)?"

#----------------------------------------------------------------------------------------------------------------------------------------------

I am building a Vision-Language Model (VLM) from scratch using a modular deep learning pipeline. The goal is to create a mini LLaVA-style architecture that takes an image + text prompt and generates a textual response.

---

# 🧠 PROJECT OVERVIEW

This project implements a multimodal system combining:

- Vision encoder (ViT)
- Query transformer (Q-Former)
- Large Language Model (Qwen)
- Projection layer for modality alignment

Pipeline:

Image → ViT → Q-Former → Projection → concat with text → LLM → output

---

# 📁 FULL PROJECT STRUCTURE

project/
│
├── config.py
│
├── models/
│ ├── vlm_model.py
│ └── qformer.py
│
├── data/
│ ├── dataset.py
│ └── collator.py
│
├── training/
│ ├── trainer.py
│ └── train.py
│
├── inference/
│ └── generate.py (planned / optional)
│
└── checkpoints/ (model saves)

---

# 📦 FILE-BY-FILE EXPLANATION

---

## 🔧 config.py

Central configuration file.

Contains:

- model names (ViT, Qwen)
- hyperparameters (lr, batch size)
- Q-Former settings
- training parameters

---

## 🧠 models/qformer.py

Implements Q-Former.

Responsibilities:

- Learnable query tokens
- Transformer encoder
- Extract important visual features from ViT output

Acts as:
→ bridge between vision and language

---

## 🧠 models/vlm_model.py

Core multimodal model.

Components:

- ViT (frozen)
- Q-Former (trainable)
- Projection layer
- Qwen LLM

Key operations:

- image → embeddings
- query extraction
- projection to LLM space
- concatenation with text embeddings
- forward pass into LLM

Includes:

- normalization
- scaling
- clamping (for stability)

---

## 📦 data/dataset.py

Loads dataset from:
→ nvidia/Llama-Nemotron-VLM-Dataset-v1

Responsibilities:

- load splits (vqa, captioning)
- filter invalid images
- return structured data:
  {
  image,
  conversation
  }

---

## 🧠 data/collator.py (CRITICAL)

Most important part for training quality.

Responsibilities:

- process images → pixel_values
- tokenize conversations using:
  tokenizer.apply_chat_template()
- create:
  - input_ids
  - attention_mask
  - labels

Label logic:

- mask all tokens (-100)
- unmask only assistant response tokens

Ensures:
→ correct instruction tuning

---

## ⚙️ training/trainer.py

Handles training step.

Responsibilities:

- forward pass
- loss computation
- backward pass
- gradient clipping
- optimizer step
- NaN handling

Important:

- NO AMP (disabled for stability)
- FP32 training

---

## 🚀 training/train.py

Entry point.

Responsibilities:

- initialize config
- load dataset + dataloader
- initialize model
- optimizer setup
- training loop

---

## 🔮 inference/generate.py (planned)

Used for testing model.

Responsibilities:

- load trained model
- process image + prompt
- generate output using LLM

---

## 💾 checkpoints/

Stores trained weights.

---

# 🏗️ MODEL ARCHITECTURE

1. Vision Encoder
   - ViT (frozen)
   - outputs patch embeddings

2. Q-Former
   - learns query tokens
   - extracts useful visual info

3. Projection Layer
   - aligns vision → language space
   - includes normalization + scaling

4. LLM (Qwen)
   - receives:
     image tokens + text tokens
   - generates response

---

# 📊 TRAINING PIPELINE

- FP32 only (no mixed precision)
- gradient clipping
- NaN batch skipping
- AdamW optimizer

---

# 📉 TRAINING BEHAVIOR OBSERVED

Loss trend:

5 → 3 → 2 → 1 → 0.3 → ~0.00005

Interpretation:

✔ initial learning
✔ convergence
❌ overfitting
❌ weak supervision signal

---

# 🚨 CURRENT PROBLEM

- Loss becomes too small
- Model memorizes dataset
- Not generalizing

---

# 🎯 CURRENT GOALS

Next improvements:

1. Add LoRA fine-tuning
2. Improve masking strategy
3. Increase dataset size
4. Add validation + early stopping
5. Improve inference quality
6. Better multimodal alignment

---

# 🧠 KEY INSIGHTS

- Low loss ≠ good model
- Data + labels > architecture
- Stability is hardest part of VLM
- Most bugs were:
  - dtype mismatch
  - AMP instability
  - incorrect masking

---

# 🚀 CURRENT STAGE

✔ Stable training system
✔ Working VLM pipeline
❌ Needs improvement in generalization

---

# 🎯 NEXT SESSION EXPECTATION

Continue from this point.

Focus on:

- improving model quality
- not debugging basics

---

# 🧠 ASSISTANT ROLE

- Understand full architecture
- Suggest real VLM improvements
- Avoid naive fixes
- Focus on system-level reasoning

---

# 🚀 FINAL CONTEXT

This is a full custom VLM system.

Goal:
→ move from “working model”
→ to “useful intelligent system”

---

Continue directly without re-explaining basics.
