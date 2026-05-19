"""Top-level multi-task network: segmentation + classification with priors."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import ClassificationHead, DecoderBlock, Downsample, EncoderBlock, SegmentationHead, Upsample
from .priors import CrossModalFusion, DynamicsPrior, ParameterSharingPrior, TopologyAwarePrior


class TiantanGliomaNet(nn.Module):
    """Multi-modal multi-task network. The classification head answers the
    main clinical question (recurrence vs radiation necrosis); the segmentation
    head provides an auxiliary lesion mask that anchors training."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 2,
        seg_classes: int = 2,
        base_channels: int = 32,
        deep_supervision: bool = True,
        use_priors: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.deep_supervision = deep_supervision
        self.use_priors = use_priors

        c = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

        self.cmf = CrossModalFusion(num_modalities=in_channels, out_channels=c[0])
        self.stem = nn.Sequential(
            nn.Conv3d(c[0], c[0], 3, padding=1, bias=False),
            nn.GroupNorm(max(1, c[0] // 8), c[0]),
            nn.GELU(),
        )

        if use_priors:
            self.ps = ParameterSharingPrior(in_channels=c[0], out_channels=c[1], num_modalities=in_channels)
        else:
            self.ps = nn.Conv3d(c[0], c[1], 3, padding=1)

        self.enc1 = EncoderBlock(c[1], c[1])
        self.down1 = Downsample(c[1], c[2])
        self.topo = TopologyAwarePrior(c[2]) if use_priors else nn.Identity()
        self.enc2 = EncoderBlock(c[2], c[2])
        self.down2 = Downsample(c[2], c[3])
        self.enc3 = EncoderBlock(c[3], c[3])
        self.down3 = Downsample(c[3], c[3])
        self.enc4 = EncoderBlock(c[3], c[3])
        self.dyn = DynamicsPrior(c[3]) if use_priors else nn.Identity()
        self.bottleneck = EncoderBlock(c[3], c[3])

        self.up3 = Upsample(c[3], c[3])
        self.dec3 = DecoderBlock(c[3] + c[3], c[3])
        self.up2 = Upsample(c[3], c[2])
        self.dec2 = DecoderBlock(c[2] + c[2], c[2])
        self.up1 = Upsample(c[2], c[1])
        self.dec1 = DecoderBlock(c[1] + c[1], c[1])

        self.seg_head = SegmentationHead(c[1], seg_classes)
        self.cls_head = ClassificationHead(c[3], num_classes)

        if deep_supervision:
            self.aux2 = SegmentationHead(c[2], seg_classes)
            self.aux3 = SegmentationHead(c[3], seg_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_aux: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        if return_aux is None:
            return_aux = self.training and self.deep_supervision
        B, M, D, H, W = x.shape
        target_size = (D, H, W)

        fused = self.cmf(x)
        stem = self.stem(fused)
        if self.use_priors:
            stem_multi = torch.stack([stem] * M, dim=1)
            enc1 = self.ps(stem_multi)
        else:
            enc1 = self.ps(stem)
        enc1 = self.enc1(enc1)

        x2 = self.down1(enc1)
        x2 = self.topo(x2)
        enc2 = self.enc2(x2)

        x3 = self.down2(enc2)
        enc3 = self.enc3(x3)

        x4 = self.down3(enc3)
        enc4 = self.enc4(x4)
        enc4 = self.dyn(enc4)
        bot = self.bottleneck(enc4)

        logits_cls = self.cls_head(bot)

        u3 = self.up3(bot)
        d3 = self.dec3(torch.cat([u3, enc3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, enc2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, enc1], dim=1))

        seg_main = self.seg_head(d1, target_size=target_size)
        outputs: Dict[str, torch.Tensor] = {"seg": seg_main, "cls": logits_cls}
        if return_aux:
            outputs["seg_aux"] = [self.aux3(d3, target_size), self.aux2(d2, target_size)]
        return outputs


def build_model(config: Dict) -> TiantanGliomaNet:
    m_cfg = config.get("model", {})
    return TiantanGliomaNet(
        in_channels=m_cfg.get("in_channels", 4),
        num_classes=m_cfg.get("num_classes", 2),
        seg_classes=m_cfg.get("seg_classes", 2),
        base_channels=m_cfg.get("base_channels", 32),
        deep_supervision=m_cfg.get("deep_supervision", True),
        use_priors=m_cfg.get("use_prior_modules", True),
    )
