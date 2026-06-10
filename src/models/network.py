"""Top-level network — BrainTTNet.

Wires::

    image  →  ModalityCouplingPrior  →  UNetBackbone(enc) ──┐
                                                            │
                                            TopologyShapePrior @ bottleneck
                                            AnatomySpatialPrior  @ bottleneck
                                                            │
                                                       backbone(dec)
                                                            │
                                                       ┌────┴────┐
                                              NestedSegHead  ClassificationHead
                                              (3 sigmoid    (rec vs nec)
                                               WT,TC,ET)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import (
    ClassificationHead,
    NestedSegmentationHead,
    UNetBackbone,
)
from .priors import (
    AnatomySpatialPrior,
    ModalityCouplingPrior,
    TopologyShapePrior,
)


class BrainTTNet(nn.Module):
    """Multi-task 3-D network for glioma recurrence vs radiation necrosis.

    Forward returns a dict:
      - ``seg_main``   : (B, 3, D, H, W) raw logits, channels [WT, TC, ET]
      - ``seg_aux``    : list of 1–2 lower-resolution seg logits when
                         deep supervision is enabled
      - ``cls``        : (B, 2) class logits
      - ``chi_pred``   : (B,) Euler characteristic estimate (topology loss)
      - ``fusion_attn``: (M,) modality fusion weights (diagnostic)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        seg_classes: int = 3,
        base_channels: int = 32,
        stem_channels: Optional[int] = None,
        deep_supervision: bool = True,
        use_modality_prior: bool = True,
        use_topology_prior: bool = True,
        use_anatomy_prior: bool = True,
        aux_dim: int = 4,
        modality_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.deep_supervision = deep_supervision
        self.modality_dropout_p = modality_dropout_p
        self.use_modality_prior = use_modality_prior
        self.use_topology_prior = use_topology_prior
        self.use_anatomy_prior = use_anatomy_prior

        stem_c = stem_channels if stem_channels is not None else base_channels

        if use_modality_prior:
            self.modality_prior = ModalityCouplingPrior(
                num_modalities=in_channels,
                per_modality_channels=8,
                out_channels=stem_c,
            )
        else:
            self.modality_prior = nn.Conv3d(in_channels, stem_c, 3, padding=1)

        self.backbone = UNetBackbone(stem_channels=stem_c, base_channels=base_channels)
        c1, c2, c3, c4 = self.backbone.channels

        self.topology_prior = TopologyShapePrior(c4) if use_topology_prior else nn.Identity()
        self.anatomy_prior = AnatomySpatialPrior(c4) if use_anatomy_prior else nn.Identity()

        self.seg_head = NestedSegmentationHead(c1, num_classes=seg_classes)
        if deep_supervision:
            self.seg_aux2 = NestedSegmentationHead(c2, num_classes=seg_classes)
            self.seg_aux3 = NestedSegmentationHead(c3, num_classes=seg_classes)

        self.cls_head = ClassificationHead(
            in_c=c4, num_classes=num_classes, aux_dim=aux_dim, dropout=0.2,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def encode(self, image: torch.Tensor,
               missing_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Returns all encoder/decoder feature maps + prior side outputs."""
        if self.training and self.modality_dropout_p > 0:
            image = ModalityCouplingPrior.modality_dropout(
                image, p=self.modality_dropout_p, keep_anchor=0,
            )

        if self.use_modality_prior:
            mod_out = self.modality_prior(image, missing_mask=missing_mask)
            stem_feat = mod_out["feat"]
            fusion_attn = mod_out.get("fusion_attn")
        else:
            stem_feat = self.modality_prior(image)
            fusion_attn = None

        feats = self.backbone(stem_feat)
        bot = feats["bottleneck"]

        chi_pred = None
        if self.use_topology_prior:
            topo_out = self.topology_prior(bot)
            bot = topo_out["feat"]
            chi_pred = topo_out.get("chi_pred")
        if self.use_anatomy_prior:
            anatomy_out = self.anatomy_prior(bot)
            bot = anatomy_out["feat"]
        feats["bottleneck"] = bot

        return {
            "feats": feats,
            "fusion_attn": fusion_attn,
            "chi_pred": chi_pred,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        missing_mask: Optional[torch.Tensor] = None,
        aux_features: Optional[torch.Tensor] = None,
        return_aux: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if return_aux is None:
            return_aux = self.training and self.deep_supervision

        target_size = image.shape[2:]
        enc_out = self.encode(image, missing_mask=missing_mask)
        feats = enc_out["feats"]

        cls_logits = self.cls_head(feats["bottleneck"], aux=aux_features)
        seg_main = self.seg_head(feats["dec1"], target_size=target_size)

        out: Dict[str, torch.Tensor] = {
            "seg": seg_main,
            "cls": cls_logits,
            "chi_pred": enc_out.get("chi_pred"),
            "fusion_attn": enc_out.get("fusion_attn"),
        }
        if return_aux and self.deep_supervision:
            out["seg_aux"] = [
                self.seg_aux3(feats["dec3"], target_size=target_size),
                self.seg_aux2(feats["dec2"], target_size=target_size),
            ]
        return out


def build_model(config: Dict) -> BrainTTNet:
    m_cfg = config.get("model", {})
    return BrainTTNet(
        in_channels=m_cfg.get("in_channels", 4),
        num_classes=m_cfg.get("num_classes", 2),
        seg_classes=m_cfg.get("seg_classes", 3),
        base_channels=m_cfg.get("base_channels", 32),
        stem_channels=m_cfg.get("stem_channels"),
        deep_supervision=m_cfg.get("deep_supervision", True),
        use_modality_prior=m_cfg.get("use_modality_prior", True),
        use_topology_prior=m_cfg.get("use_topology_prior", True),
        use_anatomy_prior=m_cfg.get("use_anatomy_prior", True),
        aux_dim=m_cfg.get("aux_dim", 4),
        modality_dropout_p=m_cfg.get("modality_dropout_p", 0.0),
    )
