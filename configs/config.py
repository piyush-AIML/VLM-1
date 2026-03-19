class Config:
    vit_model="google/vit-base-patch16-224"
    llm_model="Qwen/Qwen2-0.5B"

    num_query_tokens=32
    qformer_layers=6

    batch_size=2
    lr=1e-4
    epochs=3

    device="cuda"
    max_length=512
