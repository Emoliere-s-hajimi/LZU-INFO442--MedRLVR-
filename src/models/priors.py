"""Medical-prior modules — three consolidated blocks, each driven by a
specific subset of the M2 findings.

  1. ``ModalityCouplingPrior``  (front stem; before the U-Net encoder)
       Findings 1, 7, 9. Encodes the four modalities with shared+coupled
       weights, fuses them via attention initialised by clinical
       importance, and exposes a `modality_dropout` augmentation.

  2. ``TopologyShapePrior``      (deepest encoder feature map)
       Findings 4, 5. Multi-scale morphological-gradient attention plus a
       Euler-χ regulariser head that predicts a scalar χ proxy from the
       feature map for the topology loss term.

  3. ``AnatomySpatialPrior``     (between bottleneck and decoder)
       Findings 10, 13. Combines (a) a learnable centre-biased spatial
       prior mask that hard-zeros the 5-mm skull shell, and (b) a single
       step of reaction-diffusion dynamics that nudges the bottleneck
       latent toward biologically plausible growth fields.

Each prior follows the same contract: ``forward(x, **kwargs) -> Dict``
with at minimum a ``feat`` tensor of the same shape as the input. Extra
keys (e.g. ``chi_pred``, ``fusion_attn``) are passed back so the loss /
diagnostics can read them.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, channels // 8), channels)


# ---------------------------------------------------------------------------
# 1. Modality coupling prior
# ---------------------------------------------------------------------------

class ModalityCouplingPrior(nn.Module):
    """Front-stem prior for the 4 input modalities.

    Pipeline:
      1. per-modality CNN encoder (8 channels each)
      2. clinical coupling matrix (initialised from radiology priors —
         T1 ↔ T1ce strong, T2 ↔ FLAIR strong, cross-family weak)
      3. attention over modalities initialised to the inside-vs-outside
         separation σ ranking (FLAIR > T2 > T1ce > T1, Finding 7)
      4. projection to ``out_channels`` ready for the U-Net stem
    """

    # Clinical default — first row/col is t1, t1ce, t2, flair (canonical order)
    DEFAULT_COUPLING = torch.tensor([
        [1.0, 0.9, 0.3, 0.3],   # T1
        [0.9, 1.0, 0.4, 0.4],   # T1ce
        [0.3, 0.4, 1.0, 0.8],   # T2
        [0.3, 0.4, 0.8, 1.0],   # FLAIR
    ])
    # Modality fusion attention initialised by Finding 7 separation σ.
    DEFAULT_FUSION_WEIGHTS = torch.tensor([0.10, 0.20, 0.30, 0.40])  # T1, T1ce, T2, FLAIR

    def __init__(
        self,
        num_modalities: int = 4,
        per_modality_channels: int = 8,
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.per_modality_channels = per_modality_channels
        c = per_modality_channels

        # 4 separate stems — keeps modality-specific low-level features distinct
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, c, 3, padding=1, bias=False),
                _gn(c), nn.GELU(),
                nn.Conv3d(c, c, 3, padding=1, bias=False),
                _gn(c),
            )
            for _ in range(num_modalities)
        ])

        # Clinical coupling — initialised, then learnable
        coupling = self.DEFAULT_COUPLING[:num_modalities, :num_modalities].clone()
        self.coupling = nn.Parameter(coupling)

        # Modality fusion attention — learnable, initialised by Finding 7
        fusion = self.DEFAULT_FUSION_WEIGHTS[:num_modalities].clone()
        self.fusion_weights = nn.Parameter(fusion)

        # Per-voxel gating from coupled features (per-modality spatial mask)
        self.gate = nn.Sequential(
            nn.Conv3d(num_modalities * c, max(num_modalities * 2, 8), 1),
            _gn(max(num_modalities * 2, 8)),
            nn.GELU(),
            nn.Conv3d(max(num_modalities * 2, 8), num_modalities, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv3d(num_modalities * c, out_channels, 1, bias=False),
            _gn(out_channels),
            nn.GELU(),
        )

    @staticmethod
    def modality_dropout(image: torch.Tensor, p: float = 0.15,
                         keep_anchor: int = 0) -> torch.Tensor:
        """Bernoulli channel dropout. ``keep_anchor`` is the channel index
        that is never dropped (T1 by convention, Finding 1 — T1 is 100%
        covered in the cohort)."""
        if not torch.is_tensor(image) or p <= 0:
            return image
        B, M, *_ = image.shape
        drop = (torch.rand(B, M, 1, 1, 1, device=image.device) < p).float()
        drop[:, keep_anchor] = 0.0
        keep = 1.0 - drop
        return image * keep

    def forward(self, image: torch.Tensor,
                missing_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """``image``: (B, M, D, H, W). ``missing_mask``: optional bool (B, M)
        true where a modality is genuinely missing (zero channel from the
        cleaning pipeline). Missing channels are also forced to zero in the
        fusion weights so the model does not lean on them."""
        B, M = image.shape[:2]

        # 1. per-modality CNN
        feats = [self.encoders[m](image[:, m:m + 1]) for m in range(M)]
        # 2. clinical coupling: linearly recombine modality features
        coupling = F.softmax(self.coupling, dim=1)
        coupled = []
        for i in range(M):
            mix = sum(coupling[i, j] * feats[j] for j in range(M))
            coupled.append(mix)
        concat = torch.cat(coupled, dim=1)  # (B, M*c, D, H, W)

        # 3. spatial × modality attention
        spatial_gate = self.gate(concat)  # (B, M, D, H, W)
        global_weight = F.softmax(self.fusion_weights, dim=0).view(1, M, 1, 1, 1)
        if missing_mask is not None:
            present = (~missing_mask).float().view(B, M, 1, 1, 1)
            global_weight = global_weight * present
            # Re-normalise so present-modality weights still sum to 1
            global_weight = global_weight / (global_weight.sum(dim=1, keepdim=True) + 1e-6)

        attn = spatial_gate * global_weight
        attended = []
        c = self.per_modality_channels
        for i in range(M):
            slc = concat[:, i * c:(i + 1) * c]
            attended.append(slc * attn[:, i:i + 1])
        gated = torch.cat(attended, dim=1)

        return {
            "feat": self.proj(gated),
            "fusion_attn": global_weight.squeeze(),
            "coupling": coupling.detach(),
        }


# ---------------------------------------------------------------------------
# 2. Topology + shape prior
# ---------------------------------------------------------------------------

class TopologyShapePrior(nn.Module):
    """Topology-aware attention + Euler-χ regression head.

    The morphological gradient highlights lesion boundaries at three
    scales (3/5/7 voxels); the resulting "shape signature" is mixed back
    into the feature map via a spatial attention gate.

    A side head predicts a scalar χ proxy per case (Finding 5 — class
    prior χ_recur = +4, χ_necr = −24), used by ``TopologyChiRegulariser``
    in the loss.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(3))
        self.morph = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            _gn(channels), nn.GELU(),
        )
        # Spatial attention conditioned on the morphological signature
        sig_c = max(8, channels // 16)
        self.signature = nn.Sequential(
            nn.Conv3d(channels, sig_c, 1, bias=False), _gn(sig_c), nn.GELU(),
            nn.Conv3d(sig_c, 3, 1), nn.Sigmoid(),
        )
        self.attn = nn.Sequential(
            nn.Conv3d(channels + 3, sig_c, 1, bias=False), _gn(sig_c), nn.GELU(),
            nn.Conv3d(sig_c, 1, 1), nn.Sigmoid(),
        )
        # Euler-χ regression head — single scalar per batch element
        self.chi_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(channels, max(16, channels // 4)),
            nn.GELU(),
            nn.Linear(max(16, channels // 4), 1),
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def _morph_gradient(self, x: torch.Tensor) -> torch.Tensor:
        grads = []
        for s in (3, 5, 7):
            pad = s // 2
            dilated = F.max_pool3d(x, s, stride=1, padding=pad)
            eroded = -F.max_pool3d(-x, s, stride=1, padding=pad)
            grads.append(dilated - eroded)
        w = F.softmax(self.scale_weights, dim=0)
        return sum(wi * g for wi, g in zip(w, grads))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        identity = x
        m = self._morph_gradient(self.morph(x))
        signature = self.signature(m)                       # (B, 3, D, H, W)
        attn = self.attn(torch.cat([x, signature], dim=1))  # (B, 1, D, H, W)
        attended = x * attn
        chi_pred = self.chi_head(attended).squeeze(-1)      # (B,)
        return {
            "feat": identity + self.gamma * attended,
            "chi_pred": chi_pred,
            "shape_signature": signature,
        }


# ---------------------------------------------------------------------------
# 3. Anatomy spatial prior
# ---------------------------------------------------------------------------

class AnatomySpatialPrior(nn.Module):
    """Centre-biased spatial mask + reaction-diffusion dynamics.

    The spatial bias is a parametric "centre-up" mask: weight 1.5 in the
    central [0.3, 0.7]³ box (where 67.5% of WT centroids live, Finding
    10) and weight 0 in the outermost 5-mm shell. The bias is
    multiplicative on the bottleneck features.

    One reaction-diffusion step on top of that gives a soft prior on
    biologically plausible growth/spread fields (Finding 13 — paves the
    way for the longitudinal head in M3).
    """

    def __init__(self, channels: int, central_box: Tuple[float, float] = (0.3, 0.7),
                 shell_skip_frac: float = 0.05) -> None:
        super().__init__()
        self.channels = channels
        self.central_box = central_box
        self.shell_skip_frac = shell_skip_frac

        # Reaction-diffusion: Laplacian (channel-wise depthwise) + scalar dt
        self.dt = nn.Parameter(torch.tensor(0.1))
        self.diffusion = nn.Conv3d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)
        with torch.no_grad():
            w = torch.full((channels, 1, 3, 3, 3), -1.0 / 26.0)
            w[:, 0, 1, 1, 1] = 1.0
            self.diffusion.weight.copy_(w)
        # Per-case reaction coefficient and diffusion gain
        self.coeff_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(channels, 32), nn.GELU(),
            nn.Linear(32, 2), nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False), _gn(channels),
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def _spatial_bias(self, shape: Tuple[int, int, int], device) -> torch.Tensor:
        D, H, W = shape
        lin = lambda n: torch.linspace(0.0, 1.0, n, device=device)
        zz, yy, xx = torch.meshgrid(lin(D), lin(H), lin(W), indexing="ij")
        lo, hi = self.central_box
        central = ((zz >= lo) & (zz <= hi) & (yy >= lo) & (yy <= hi) & (xx >= lo) & (xx <= hi)).float()
        bias = 1.0 + 0.5 * central
        # Hard zero the outermost shell along each axis (skull-adjacent)
        s = self.shell_skip_frac
        skull = ((zz < s) | (zz > 1 - s) | (yy < s) | (yy > 1 - s) | (xx < s) | (xx > 1 - s)).float()
        bias = bias * (1.0 - skull)
        return bias.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        identity = x
        bias = self._spatial_bias(x.shape[2:], x.device)
        biased = x * bias

        # Reaction-diffusion step
        params = self.coeff_head(biased)  # (B, 2)
        rho = params[:, 0].view(-1, 1, 1, 1, 1)
        diff = params[:, 1].view(-1, 1, 1, 1, 1)
        lap = self.diffusion(biased) - biased
        u = torch.sigmoid(biased)
        reaction = u * (1 - u)
        dt = torch.clamp(self.dt, 0.01, 0.5)
        evolved = biased + dt * (diff * lap + rho * reaction)

        return {
            "feat": identity + self.gamma * self.proj(evolved),
            "spatial_bias": bias.squeeze(),
            "rd_coeffs": params,
        }
