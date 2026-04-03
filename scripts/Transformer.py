import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, c, num_heads=8, depth=3, mlp_ratio=4.0):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(c),
                nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, dropout=0.1, batch_first=True),
                nn.LayerNorm(c),
                nn.Sequential(
                    nn.Linear(c, int(c * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(int(c * mlp_ratio), c),
                    nn.Dropout(0.1)
                )
            ]))

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        for norm1, attn, norm2, mlp in self.layers:
            residual = x_flat
            x_flat = norm1(x_flat)
            attn_out, _ = attn(x_flat, x_flat, x_flat)
            x_flat = residual + attn_out

            residual = x_flat
            x_flat = norm2(x_flat)
            x_flat = residual + mlp(x_flat)

        x = x_flat.transpose(1, 2).reshape(b, c, h, w)
        return x
