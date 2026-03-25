import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 🔁 Feed Forward (Gated MLP - SOTA style)
# ----------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden = dim * expansion

        self.fc1 = nn.Linear(dim, hidden * 2)  # for gated activation
        self.fc2 = nn.Linear(hidden, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_proj = self.fc1(x)
        x, gate = x_proj.chunk(2, dim=-1)

        x = x * F.silu(gate)  # 🔥 gated activation (better than GELU)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ----------------------------
# ⚡ SOTA Attention (SDPA + temp scaling)
# ----------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        assert dim % heads == 0

        self.heads = heads
        self.head_dim = dim // heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim)

        self.dropout = dropout

        # 🔥 learnable temperature (important)
        self.scale = nn.Parameter(torch.ones(heads))

    def forward(self, q, k, v):
        B, N, D = q.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, k.size(1), self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, v.size(1), self.heads, self.head_dim).transpose(1, 2)

        # 🔥 scale queries per head
        q = q * self.scale.view(1, -1, 1, 1)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = attn.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


# ----------------------------
# 🔁 Residual Block Base (DeepNet style)
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * 0.5)  # 🔥 stabilizer
        self.dropout = nn.Dropout(dropout)

    def apply_residual(self, x, residual):
        return x + self.scale * self.dropout(residual)


# ----------------------------
# 🔁 Cross Attention Block (Upgraded)
# ----------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

        self.res1 = ResidualBlock(dim, dropout)
        self.res2 = ResidualBlock(dim, dropout)

        # 🔥 gated fusion (important)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, query, key_value):
        h = self.norm1(query)

        attn_out = self.attn(h, key_value, key_value)

        # 🔥 gated cross attention
        gated = torch.sigmoid(self.gate) * attn_out
        query = self.res1.apply_residual(query, gated)

        ff_out = self.ff(self.norm2(query))
        query = self.res2.apply_residual(query, ff_out)

        return query


# ----------------------------
# 🔁 Self Attention Block (Upgraded)
# ----------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

        self.res1 = ResidualBlock(dim, dropout)
        self.res2 = ResidualBlock(dim, dropout)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.norm1(x)

        attn_out = self.attn(h, h, h)

        gated = torch.sigmoid(self.gate) * attn_out
        x = self.res1.apply_residual(x, gated)

        ff_out = self.ff(self.norm2(x))
        x = self.res2.apply_residual(x, ff_out)

        return x


# ----------------------------
# 🔥 FINAL Q-FORMER (SOTA STYLE)
# ----------------------------
class QFormer(nn.Module):
    def __init__(self, dim=768, num_queries=32, layers=6, dropout=0.1):
        super().__init__()

        self.num_queries = num_queries

        # 🔥 better initialization
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dim) * (dim**-0.5))

        self.query_pos = nn.Parameter(torch.zeros(1, num_queries, dim))

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
