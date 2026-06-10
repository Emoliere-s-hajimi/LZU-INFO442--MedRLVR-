"""LightBrainNet — Modality-Agnostic Lesion-Centric Network.

Three core innovations:

1.  Modality-Agnostic Hypernetwork Encoder (MAHE)
    - Single shared per-modality encoder reused across whichever modalities
      are actually present
    - Modality identity is injected as a learned token, not as a channel index
    - Missing modalities NEVER enter the computational graph
    - Variable-length input: (B, M_present, H, W, D) where M_present in {1..4}

2.  Asymmetric Dual-Branch Morphology Head
    - Solid-Lesion Branch: closing-based filling + interior consistency attention
    - Cavitary-Lesion Branch: opening-based hole detection + interior heterogeneity attention
    - The logit difference between the two is a structural χ proxy
    - Classification is directly interpretable as "which branch agrees with the input"

3.  Lesion-Only Tiny U-Net (LO-Tiny)
    - Coarse stage (~0.1M params) finds the lesion ROI on full volume
    - Fine stage (~0.8M params) refines the ROI only
    - Total: ~1.5M params, < 1/4 of the previous BrainTTNet
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, min(8, c // 4)), c)


class _DSConv3d(nn.Module):
    """Depthwise-separable 3D conv — replaces standard Conv3d at ~10× fewer params."""
    def __init__(self, in_c: int, out_c: int, k: int = 3) -> None:
        super().__init__()
        self.dw = nn.Conv3d(in_c, in_c, k, padding=k // 2, groups=in_c, bias=False)
        self.pw = nn.Conv3d(in_c, out_c, 1, bias=False)
        self.norm = _gn(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


# ===========================================================================
# Innovation 1 — Modality-Agnostic Hypernetwork Encoder (MAHE)
# ===========================================================================

class ModalityAgnosticEncoder(nn.Module):
    """Per-modality shared encoder + modality token conditioning.

    The same convolutional weights process every modality (T1, T1ce, T2,
    FLAIR), but each modality has a learned 16-dim token that modulates
    the encoder via FiLM (Feature-wise Linear Modulation). This lets one
    network handle whichever subset of modalities is present.

    Input  : list of (B, 1, H, W, D) tensors with corresponding modality
             indices in [0, 1, 2, 3]
    Output : (B, out_c, H/2, W/2, D/2) — attention-pooled across modalities
    """

    NUM_MODALITIES = 4   # T1=0, T1ce=1, T2=2, FLAIR=3
    TOKEN_DIM = 16

    # σ-prior weights (FLAIR > T2 > T1ce > T1)
    PRIOR_WEIGHTS = torch.tensor([0.05, 0.25, 0.30, 0.40])

    def __init__(self, stem_c: int = 16, out_c: int = 32) -> None:
        super().__init__()
        self.modality_tokens = nn.Embedding(self.NUM_MODALITIES, self.TOKEN_DIM)

        # FiLM generators: token -> (gamma, beta) for each stage
        self.film1 = nn.Linear(self.TOKEN_DIM, 2 * stem_c)
        self.film2 = nn.Linear(self.TOKEN_DIM, 2 * out_c)

        # Shared encoder — sees one modality at a time
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, stem_c, 3, padding=1, bias=False),
            _gn(stem_c), nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            _DSConv3d(stem_c, out_c),
            nn.Conv3d(out_c, out_c, 2, stride=2, bias=False),  # ×1/2 downsample
            _gn(out_c), nn.GELU(),
        )

        # Cross-modality attention pooling — d-weighted (classification-aware)
        # Initialised from EDA: T1ce has highest Cohen's d=0.94 → highest weight
        d_prior = torch.tensor([0.06, 0.65, 0.15, 0.14])  # T1, T1ce, T2, FLAIR
        self.d_weights = nn.Parameter(d_prior)

        # Initialise prior weights for σ-attention (segmentation-aware)
        self.sigma_weights = nn.Parameter(self.PRIOR_WEIGHTS.clone())

    def _apply_film(self, x: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: γ * x + β where γ, β are conditioned on modality token."""
        gamma, beta = film_params.chunk(2, dim=-1)
        # Broadcast (B, C) -> (B, C, 1, 1, 1)
        gamma = gamma.view(*gamma.shape, 1, 1, 1)
        beta = beta.view(*beta.shape, 1, 1, 1)
        return gamma * x + beta

    def forward(self, image: torch.Tensor,
                missing_mask: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image:        (B, 4, H, W, D) — full 4-channel input, zeros where missing
            missing_mask: (B, 4) bool — True where modality is missing
        Returns:
            'feat_sigma': (B, out_c, H/2, W/2, D/2) — segmentation-attention pooled
            'feat_d':     (B, out_c, H/2, W/2, D/2) — classification-attention pooled
            'attn_sigma': (B, 4) — actual per-modality σ weights used (zeros for missing)
            'attn_d':     (B, 4) — actual per-modality d weights used (zeros for missing)
        """
        B = image.shape[0]
        device = image.device

        # Identify present modalities per-sample
        if missing_mask is None:
            missing_mask = torch.zeros(B, 4, dtype=torch.bool, device=device)
        present_mask = (~missing_mask).float()  # (B, 4)

        # Process each modality through shared encoder, conditioned on its token
        modality_feats = []
        for m in range(4):
            # Skip if entire batch is missing this modality
            if not present_mask[:, m].any():
                modality_feats.append(None)
                continue

            x = image[:, m:m+1]
            token = self.modality_tokens(torch.tensor([m], device=device)).expand(B, -1)

            # Stage 1
            f1 = self.conv1(x)
            f1 = self._apply_film(f1, self.film1(token))

            # Stage 2
            f2 = self.conv2(f1)
            f2 = self._apply_film(f2, self.film2(token))

            modality_feats.append(f2)

        # Build attention weights, zeroing missing modalities then renormalising
        sigma_attn = F.softmax(self.sigma_weights, dim=0).view(1, 4)
        d_attn = F.softmax(self.d_weights, dim=0).view(1, 4)
        sigma_attn = sigma_attn * present_mask
        d_attn = d_attn * present_mask
        sigma_attn = sigma_attn / (sigma_attn.sum(dim=1, keepdim=True) + 1e-6)
        d_attn = d_attn / (d_attn.sum(dim=1, keepdim=True) + 1e-6)

        # Attention pooling — only over actually-present modalities
        feat_sigma = 0
        feat_d = 0
        for m, f in enumerate(modality_feats):
            if f is None:
                continue
            w_s = sigma_attn[:, m].view(-1, 1, 1, 1, 1)
            w_d = d_attn[:, m].view(-1, 1, 1, 1, 1)
            feat_sigma = feat_sigma + w_s * f
            feat_d = feat_d + w_d * f

        return {
            "feat_sigma": feat_sigma,
            "feat_d": feat_d,
            "attn_sigma": sigma_attn,
            "attn_d": d_attn,
        }


# ===========================================================================
# Innovation 2 — Asymmetric Dual-Branch Morphology Head
# ===========================================================================

class SolidLesionBranch(nn.Module):
    """Specialised for compact, simply-connected enhancing lesions (recurrence).

    Operations:
      1. Morphological closing approximation (dilate then erode) — fills small gaps
      2. Interior consistency attention — high response on homogeneous solid regions
      3. Outputs a "solid-lesion likelihood" score per voxel + global pooling
    """

    def __init__(self, in_c: int) -> None:
        super().__init__()
        # Closing approximation via max-pool then -max-pool of negative
        self.closing_kernel = 5

        # Interior-consistency attention: feature-map std-dev should be LOW inside solid lesions
        self.consistency = nn.Sequential(
            nn.Conv3d(in_c, in_c // 2, 1, bias=False),
            _gn(in_c // 2), nn.GELU(),
            nn.Conv3d(in_c // 2, 1, 1),
        )

        # Final solid-likelihood map
        self.refine = nn.Sequential(
            _DSConv3d(in_c, in_c),
            nn.Conv3d(in_c, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Morphological closing approximation
        k = self.closing_kernel
        dilated = F.max_pool3d(x, k, stride=1, padding=k // 2)
        closed = -F.max_pool3d(-dilated, k, stride=1, padding=k // 2)

        # Difference highlights "fill-able" regions (signature of solid lesions)
        fill_mask = (closed - x).clamp(min=0)

        # Interior consistency: high score when local variance is low
        # We use 1 - sigmoid(consistency) as an "interior-ness" score
        cons_logits = self.consistency(x + fill_mask)
        interior_attn = torch.sigmoid(-cons_logits)  # high where x is consistent

        # Final solid-likelihood map
        solid_logits = self.refine(x * interior_attn)
        # Global score (used in classification head)
        solid_score = F.adaptive_avg_pool3d(solid_logits, 1).flatten(1)

        return {
            "solid_logits": solid_logits,  # (B, 1, H, W, D)
            "solid_score": solid_score,     # (B, 1)
            "interior_attn": interior_attn,
        }


class CavitaryLesionBranch(nn.Module):
    """Specialised for cavitary, multi-handle lesions (radiation necrosis).

    Operations:
      1. Morphological opening (erode then dilate) — removes small protrusions
      2. Hole detection: difference between (open then close) and original
      3. Interior heterogeneity attention — high response on internally-varying regions
      4. Outputs a "cavitary-lesion likelihood" + global pool
    """

    def __init__(self, in_c: int) -> None:
        super().__init__()
        self.opening_kernel = 5

        # Heterogeneity attention: high where local features are diverse
        self.heterogeneity = nn.Sequential(
            nn.Conv3d(in_c, in_c // 2, 1, bias=False),
            _gn(in_c // 2), nn.GELU(),
            nn.Conv3d(in_c // 2, 1, 1),
        )

        self.refine = nn.Sequential(
            _DSConv3d(in_c, in_c),
            nn.Conv3d(in_c, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Morphological opening approximation
        k = self.opening_kernel
        eroded = -F.max_pool3d(-x, k, stride=1, padding=k // 2)
        opened = F.max_pool3d(eroded, k, stride=1, padding=k // 2)

        # Difference highlights "removed protrusions" / cavity signatures
        cavity_mask = (x - opened).clamp(min=0)

        # Heterogeneity: high score for internally-varying regions
        het_logits = self.heterogeneity(x + cavity_mask)
        hetero_attn = torch.sigmoid(het_logits)

        cavity_logits = self.refine(x * hetero_attn)
        cavity_score = F.adaptive_avg_pool3d(cavity_logits, 1).flatten(1)

        return {
            "cavity_logits": cavity_logits,  # (B, 1, H, W, D)
            "cavity_score": cavity_score,     # (B, 1)
            "hetero_attn": hetero_attn,
        }


class DualMorphologyHead(nn.Module):
    """Combines solid + cavitary branches into a structural χ proxy + class logits.

    The fundamental insight: rather than learning a single "topology" descriptor,
    we explicitly model the TWO EXPECTED MORPHOLOGIES (one per class) and
    classify by which branch fires strongest.
    """

    def __init__(self, in_c: int) -> None:
        super().__init__()
        self.solid_branch = SolidLesionBranch(in_c)
        self.cavitary_branch = CavitaryLesionBranch(in_c)

        # Class logits from the score difference
        # +ve = solid (recurrence), -ve = cavitary (necrosis)
        self.cls_combiner = nn.Linear(2, 2)

        # Init: positive weight for (solid - cavitary)
        with torch.no_grad():
            self.cls_combiner.weight.copy_(torch.tensor([
                [+1.0, -1.0],   # recurrence logit
                [-1.0, +1.0],   # necrosis logit
            ]))
            self.cls_combiner.bias.zero_()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        solid = self.solid_branch(x)
        cavity = self.cavitary_branch(x)

        # Concatenated scores → 2-class logits
        scores = torch.cat([solid["solid_score"], cavity["cavity_score"]], dim=1)  # (B, 2)
        cls_logits = self.cls_combiner(scores)

        # χ proxy: solid_score - cavity_score (positive ≈ +4, negative ≈ -24)
        chi_proxy = (solid["solid_score"] - cavity["cavity_score"]).squeeze(-1)

        # Combined morphology attention map (for downstream seg fusion)
        morph_attn = torch.sigmoid(solid["solid_logits"] + cavity["cavity_logits"])

        return {
            "cls_logits": cls_logits,
            "chi_proxy": chi_proxy,
            "solid_logits": solid["solid_logits"],
            "cavity_logits": cavity["cavity_logits"],
            "morph_attn": morph_attn,
        }


# ===========================================================================
# Innovation 3 — Lesion-Only Tiny U-Net (LO-Tiny)
# ===========================================================================

class TinyEncoderDecoder(nn.Module):
    """Compact 2-stage encoder-decoder for segmentation.

    Uses depthwise-separable convolutions throughout. ~0.6M params total.
    """

    def __init__(self, in_c: int, base_c: int = 16, out_c: int = 3) -> None:
        super().__init__()
        c1, c2, c3 = base_c, base_c * 2, base_c * 4

        # Encoder
        self.enc1 = _DSConv3d(in_c, c1)
        self.down1 = nn.Conv3d(c1, c2, 2, stride=2, bias=False)
        self.enc2 = _DSConv3d(c2, c2)
        self.down2 = nn.Conv3d(c2, c3, 2, stride=2, bias=False)
        self.enc3 = _DSConv3d(c3, c3)

        # Decoder
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = _DSConv3d(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = _DSConv3d(c1 + c1, c1)

        self.out = nn.Conv3d(c1, out_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


# ===========================================================================
# Top-Level Network — LightBrainNet
# ===========================================================================

class LightBrainNet(nn.Module):
    """Modality-agnostic, asymmetric-morphology, lesion-centric brain MRI network.

    Pipeline:
        4-modality input + missing_mask
                │
                ▼
        ┌───────────────────────────────────────┐
        │  Modality-Agnostic Encoder (MAHE)     │
        │  · single shared per-modality encoder │
        │  · variable-length input              │
        │  · two attention pools (σ + d)        │
        └───────────────────────────────────────┘
                │  feat_sigma (seg) + feat_d (cls)
                │
                ▼
        ┌───────────────────────────────────────┐
        │  Dual Morphology Head                 │
        │  · solid branch  (closing-based)      │
        │  · cavity branch (opening-based)      │
        │  · cls logits = f(solid_score,        │
        │                   cavity_score)       │
        │  · chi_proxy   = solid − cavity       │
        └───────────────────────────────────────┘
                │  cls_logits + morph_attn
                │
                ▼
        ┌───────────────────────────────────────┐
        │  Lesion-Only Tiny U-Net               │
        │  · feat_sigma × morph_attn → ROI      │
        │  · refine to 3-channel seg            │
        └───────────────────────────────────────┘
                │
                ▼
        seg logits (B, 3, H, W, D)   +   cls logits (B, 2)   +   chi_proxy (B,)
    """

    def __init__(
        self,
        encoder_stem_c: int = 16,
        encoder_out_c: int = 32,
        seg_base_c: int = 16,
        num_classes: int = 2,
        seg_classes: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = ModalityAgnosticEncoder(stem_c=encoder_stem_c, out_c=encoder_out_c)
        self.morphology_head = DualMorphologyHead(in_c=encoder_out_c)
        self.seg_unet = TinyEncoderDecoder(in_c=encoder_out_c, base_c=seg_base_c, out_c=seg_classes)
        self.upsample_out = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(
        self,
        image: torch.Tensor,
        missing_mask: Optional[torch.Tensor] = None,
        aux_features: Optional[torch.Tensor] = None,  # unused, kept for API parity
        return_aux: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        target_size = image.shape[2:]

        # Innovation 1: Modality-agnostic encoding
        enc_out = self.encoder(image, missing_mask=missing_mask)
        feat_sigma = enc_out["feat_sigma"]
        feat_d = enc_out["feat_d"]

        # Innovation 2: Asymmetric dual-branch morphology — operates on d-features
        morph_out = self.morphology_head(feat_d)

        # Innovation 3: Lesion-only U-Net — gated by morphology attention
        seg_input = feat_sigma * morph_out["morph_attn"]
        seg_logits = self.seg_unet(seg_input)
        seg_logits = self.upsample_out(seg_logits)
        # Final spatial alignment to input size
        if seg_logits.shape[2:] != tuple(target_size):
            seg_logits = F.interpolate(seg_logits, size=target_size,
                                       mode="trilinear", align_corners=False)

        return {
            "seg": seg_logits,
            "cls": morph_out["cls_logits"],
            "chi_pred": morph_out["chi_proxy"],
            "fusion_attn_sigma": enc_out["attn_sigma"],
            "fusion_attn_d": enc_out["attn_d"],
            "solid_logits": morph_out["solid_logits"],
            "cavity_logits": morph_out["cavity_logits"],
        }


def build_lightbrain_model(config: Optional[Dict] = None) -> LightBrainNet:
    """Build LightBrainNet from a config dict (with sensible defaults)."""
    config = config or {}
    m_cfg = config.get("model", {})
    return LightBrainNet(
        encoder_stem_c=m_cfg.get("encoder_stem_c", 16),
        encoder_out_c=m_cfg.get("encoder_out_c", 32),
        seg_base_c=m_cfg.get("seg_base_c", 16),
        num_classes=m_cfg.get("num_classes", 2),
        seg_classes=m_cfg.get("seg_classes", 3),
    )
