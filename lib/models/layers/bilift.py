import torch
import torch.nn as nn


class LowRankCrossUpdate(nn.Module):
    def __init__(self, dim, rank, dropout=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.xavier_normal_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, source):
        hidden = self.down(self.norm(source))
        return self.up(self.dropout(self.act(hidden)))


class BiLift(nn.Module):
    def __init__(self, dim, rank, dropout=0.0, reverse=False, diagnostics=False):
        super().__init__()
        self.reverse = reverse
        self.diagnostics = diagnostics
        self.x_to_rgb = LowRankCrossUpdate(dim, rank, dropout)
        self.rgb_to_x = LowRankCrossUpdate(dim, rank, dropout)
        self.last_stats = {}

    def forward(self, rgb, x):
        if self.reverse:
            r2x_update = self.rgb_to_x(rgb)
            x_out = x + r2x_update
            x2r_update = self.x_to_rgb(x_out)
            rgb_out = rgb + x2r_update
        else:
            x2r_update = self.x_to_rgb(x)
            rgb_out = rgb + x2r_update
            r2x_update = self.rgb_to_x(rgb_out)
            x_out = x + r2x_update
        self._record_stats(rgb, x, rgb_out, x_out, x2r_update, r2x_update)
        return rgb_out, x_out

    def inverse(self, rgb_out, x_out):
        if self.reverse:
            rgb = rgb_out - self.x_to_rgb(x_out)
            x = x_out - self.rgb_to_x(rgb)
        else:
            x = x_out - self.rgb_to_x(rgb_out)
            rgb = rgb_out - self.x_to_rgb(x)
        return rgb, x

    def _record_stats(self, rgb, x, rgb_out, x_out, x2r_update, r2x_update):
        self.last_stats = {}
        if not self.diagnostics:
            return
        with torch.no_grad():
            rgb_norm = torch.linalg.vector_norm(rgb.detach()).clamp_min(1e-6)
            x_norm = torch.linalg.vector_norm(x.detach()).clamp_min(1e-6)
            difference_norm = torch.linalg.vector_norm((rgb - x).detach()).clamp_min(1e-6)
            self.last_stats = {
                "BiLift/x2r_update_ratio": torch.linalg.vector_norm(x2r_update.detach()) / rgb_norm,
                "BiLift/r2x_update_ratio": torch.linalg.vector_norm(r2x_update.detach()) / x_norm,
                "BiLift/difference_ratio": (
                    torch.linalg.vector_norm((rgb_out - x_out).detach()) / difference_norm
                ),
            }
