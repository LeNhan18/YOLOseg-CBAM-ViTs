
import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, c, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c//reduction, c, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        ca = self.channel_att(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_att(sa)
        return x * sa
