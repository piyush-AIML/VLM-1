class Config:
    # ---------------- MODELS ----------------
    vit_model = "google/vit-base-patch16-224"
    llm_model = "Qwen/Qwen2-0.5B"

    # ---------------- Q-FORMER ----------------
    num_query_tokens = 32
    qformer_layers = 6

    # ---------------- TRAINING ----------------
    batch_size = 2
    grad_accum_steps = 4  # effective batch = 8
    use_fp16 = True
    lr = 5e-6  # 🔥 LOWER (better stability)
    weight_decay = 0.01

    epochs = 8  # 🔥 increase (3 is too low)

    # DataLoader (set to 0 for debugging on some platforms)
    num_workers_train = 4
    num_workers_val = 2

    # ---------------- SEQUENCE ----------------
    max_length = 512

    # ---------------- SYSTEM ----------------
    device = "cuda"

    # ---------------- SCHEDULER ----------------
    warmup_ratio = 0.05

    # ---------------- LOGGING ----------------
    log_every_steps = 10
    wandb_project = "vlm-project"
    wandb_run_name = "vlm-run"
    # ----------------------------
    # 🔥 LoRA CONFIG (ADD THIS)
    # ----------------------------
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05

    # None = use default in model
    lora_targets = None
    gradient_checkpointing = True
    pad_token_id = 0
