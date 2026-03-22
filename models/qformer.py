import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 🔁 Feed Forward (Scaled)
# ----------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden_dim = dim * expansion

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ----------------------------
# ⚡ Efficient Attention (SDPA-ready)
# ----------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, q, k, v):
        B, N, D = q.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # reshape → [B, heads, seq, head_dim]
        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, k.size(1), self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, v.size(1), self.heads, self.head_dim).transpose(1, 2)

        # 🔥 use PyTorch SDPA (faster, stable, AMP safe)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # reshape back
        out = attn_out.transpose(1, 2).contiguous().view(B, N, D)

        return self.out_proj(out)


# ----------------------------
# 🔁 Cross Attention Block (Gated, Stable)
# ----------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

        # 🔥 gated residual (stable training)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key_value):
        h = self.norm1(query)
        attn_out = self.attn(h, key_value, key_value)

        query = query + self.gate * attn_out
        query = query + self.ff(self.norm2(query))

        return query


# ----------------------------
# 🔁 Self Attention Block (Gated, Stable)
# ----------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        h = self.norm1(x)
        attn_out = self.attn(h, h, h)

        x = x + self.gate * attn_out
        x = x + self.ff(self.norm2(x))

        return x


# ----------------------------
# 🔥 FINAL Q-FORMER (DROP-IN)
# ----------------------------
class QFormer(nn.Module):
    def __init__(self, dim=768, num_queries=32, layers=6, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.num_queries = num_queries

        # 🔥 learnable queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)

        # 🔥 positional encoding
        self.query_pos = nn.Parameter(torch.zeros(1, num_queries, dim))

        # 🔁 alternating blocks
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(dim, dropout=dropout) for _ in range(layers)]
        )

        self.self_blocks = nn.ModuleList(
            [SelfAttentionBlock(dim, dropout=dropout) for _ in range(layers)]
        )

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, image_embeds):
        B = image_embeds.size(0)

        queries = self.query_tokens.expand(B, -1, -1)
        queries = queries + self.query_pos

        for cross, self_attn in zip(self.cross_blocks, self.self_blocks):
            queries = cross(queries, image_embeds)
            queries = self_attn(queries)

        return self.final_norm(queries)
