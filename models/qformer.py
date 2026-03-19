import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + attn_out)


class QFormer(nn.Module):
    def __init__(self, dim=768, num_queries=32, layers=6):
        super().__init__()

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, dim))

        self.cross_attn = nn.ModuleList([
            CrossAttentionBlock(dim) for _ in range(layers)
        ])

        self.self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)
            for _ in range(layers)
        ])

    def forward(self, image_embeds):
        B = image_embeds.size(0)
        queries = self.query_tokens.expand(B, -1, -1)

        for ca, sa in zip(self.cross_attn, self.self_attn):
            queries = ca(queries, image_embeds)
            queries = sa(queries)

        return queries
