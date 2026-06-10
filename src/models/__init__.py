"""Model package.

Public surface:

    * Backbone primitives:  ``ResidualBlock3D``, ``AnisotropicResidualBlock3D``,
                            ``Downsample``, ``Upsample``, ``UNetBackbone``,
                            ``NestedSegmentationHead``, ``ClassificationHead``
    * Priors (3 modules)  : ``ModalityCouplingPrior``,
                            ``TopologyShapePrior``,
                            ``AnatomySpatialPrior``
    * Top-level model     : ``BrainTTNet``, ``build_model``
"""
from .backbone import (
    AnisotropicResidualBlock3D,
    ClassificationHead,
    Downsample,
    NestedSegmentationHead,
    ResidualBlock3D,
    UNetBackbone,
    Upsample,
)
from .network import BrainTTNet, build_model
from .priors import (
    AnatomySpatialPrior,
    ModalityCouplingPrior,
    TopologyShapePrior,
)

__all__ = [
    "AnisotropicResidualBlock3D",
    "AnatomySpatialPrior",
    "BrainTTNet",
    "ClassificationHead",
    "Downsample",
    "ModalityCouplingPrior",
    "NestedSegmentationHead",
    "ResidualBlock3D",
    "TopologyShapePrior",
    "UNetBackbone",
    "Upsample",
    "build_model",
]
