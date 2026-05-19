"""Medical-prior modules used to inject domain knowledge into the network.

The three priors are placeholders at the proposal stage — concrete designs will
be refined during the implementation phase. The current implementations capture
the spirit of each prior so the rest of the pipeline can be wired end-to-end.

- `CrossModalFusion`: gated fusion of T1 / T1ce / T2 / FLAIR modalities.
- `ParameterSharingPrior`: tied-weight per-modality encoder with a learnable
  coupling matrix that reflects clinical co-occurrence between modalities.
- `TopologyAwarePrior`: morphological-gradient + Betti-style attention designed
  to preserve lesion topology.
- `DynamicsPrior`: a small reaction-diffusion step that nudges the latent
  representation toward biologically plausible tumor growth dynamics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, channels // 8), channels)


class CrossModalFusion(nn.Module):
    def __init__(self, num_modalities: int = 4, out_channels: int = 32) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(1, 8, 3, padding=1, bias=False), nn.GroupNorm(max(1, 8), 8), nn.GELU()
                )
                for _ in range(num_modalities)
            ]
        )
        self.attn = nn.Sequential(
            nn.Conv3d(num_modalities * 8, max(8, num_modalities * 2), 1),
            nn.GroupNorm(max(1, num_modalities * 2), max(8, num_modalities * 2)),
            nn.GELU(),
            nn.Conv3d(max(8, num_modalities * 2), num_modalities, 1),
            nn.Sigmoid(),
        )
        self.weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        self.proj = nn.Sequential(
            nn.Conv3d(num_modalities * 8, out_channels, 3, padding=1), _gn(out_channels), nn.GELU()
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, D, H, W = x.shape
        feats = [self.encoders[m](x[:, m : m + 1]) for m in range(M)]
        concat = torch.cat(feats, dim=1)
        attn = self.attn(concat)
        gated = [self.weights[m] * attn[:, m : m + 1] * feats[m] for m in range(M)]
        return self.gamma * self.proj(torch.cat(gated, dim=1))


class ParameterSharingPrior(nn.Module):
    """Modalities share an encoder backbone but are recombined via a learnable
    coupling matrix initialised to reflect clinical similarity (T1 ↔ T1ce,
    T2 ↔ FLAIR)."""

    def __init__(self, in_channels: int, out_channels: int, num_modalities: int = 4) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.shared = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
            _gn(in_channels),
            nn.GELU(),
        )
        coupling_init = torch.tensor(
            [
                [1.0, 0.9, 0.3, 0.3],
                [0.9, 1.0, 0.4, 0.4],
                [0.3, 0.4, 1.0, 0.8],
                [0.3, 0.4, 0.8, 1.0],
            ]
        )
        self.coupling = nn.Parameter(coupling_init[:num_modalities, :num_modalities].clone())
        self.fuse = nn.Sequential(
            nn.Conv3d(in_channels * num_modalities, out_channels, 1), _gn(out_channels), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, C, D, H, W = x.shape
        encoded = torch.stack([self.shared(x[:, m]) for m in range(M)], dim=1)
        coupling = F.softmax(self.coupling, dim=1)
        flat = encoded.permute(0, 2, 3, 4, 5, 1).reshape(B, -1, M)
        coupled = torch.bmm(flat, coupling.unsqueeze(0).expand(B, -1, -1))
        coupled = coupled.reshape(B, C, D, H, W, M).permute(0, 5, 1, 2, 3, 4)
        return self.fuse(coupled.reshape(B, M * C, D, H, W))


class TopologyAwarePrior(nn.Module):
    """Multi-scale morphological gradient + topology-aware spatial attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(3))
        self.morph = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False), _gn(channels), nn.GELU()
        )
        self.topo_enc = nn.Sequential(
            nn.Conv3d(channels, max(16, channels // 8), 1),
            _gn(max(16, channels // 8)),
            nn.GELU(),
            nn.Conv3d(max(16, channels // 8), 3, 1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(channels + 3, max(8, channels // 16), 1),
            _gn(max(8, channels // 16)),
            nn.GELU(),
            nn.Conv3d(max(8, channels // 16), 1, 1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv3d(channels, channels, 1),
            _gn(channels),
        )
        self.gamma = nn.Parameter(torch.ones(1))
        self.residual_scale = nn.Parameter(torch.ones(1))

    def _morph_gradient(self, x: torch.Tensor) -> torch.Tensor:
        grads = []
        for scale in (3, 5, 7):
            pad = scale // 2
            d = F.max_pool3d(x, scale, stride=1, padding=pad)
            e = -F.max_pool3d(-x, scale, stride=1, padding=pad)
            grads.append(d - e)
        w = F.softmax(self.scale_weights, dim=0)
        return sum(wi * g for wi, g in zip(w, grads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        m = self._morph_gradient(self.morph(x))
        topo = self.topo_enc(m)
        attn = self.spatial_attn(torch.cat([x, topo], dim=1))
        return self.gamma * self.out_proj(x * attn) + self.residual_scale * identity


class DynamicsPrior(nn.Module):
    """One discrete step of reaction-diffusion, used as a soft prior on the
    spatial evolution of glioma vs necrosis patterns."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.dt = nn.Parameter(torch.tensor(0.1))
        self.physio = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )
        self.laplacian = nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self._init_laplacian()
        self.spatial_weight = nn.Sequential(
            nn.Conv3d(channels, max(8, channels // 16), 1),
            _gn(max(8, channels // 16)),
            nn.GELU(),
            nn.Conv3d(max(8, channels // 16), 1, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(nn.Conv3d(channels, channels, 1), _gn(channels))
        self.gamma = nn.Parameter(torch.ones(1))

    def _init_laplacian(self) -> None:
        w = torch.full((1, 1, 3, 3, 3), -1.0 / 6.0)
        w[0, 0, 1, 1, 1] = 1.0
        with torch.no_grad():
            self.laplacian.weight.data = w.repeat(self.channels, 1, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        params = self.physio(x)
        rho = params[:, 0].view(-1, 1, 1, 1, 1)
        diff = params[:, 1].view(-1, 1, 1, 1, 1)
        lap = self.laplacian(x) - x
        u = torch.sigmoid(x)
        reaction = u * (1 - u)
        dt = torch.clamp(self.dt, 0.01, 0.5)
        evolved = x + dt * (diff * lap + rho * reaction)
        w = self.spatial_weight(evolved)
        modulated = x + w * (evolved - x)
        return self.gamma * self.proj(modulated) + identity
