"""
MobNetFine: MobileNetV3-Large with Multi-Scale Attention Feature Fusion (MSAF).

The model reuses ImageNet-pretrained MobileNetV3-Large feature stages and
replaces its original classifier with a compact multi-scale attention head.

Input : [B, 3, H, W] (normally 224 x 224)
Output: [B, num_classes] raw logits
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)

from src.data.class_names import NUM_CLASSES


class ChannelSpatialAttention(nn.Module):
    """CBAM-style channel and spatial attention for a 2D feature map."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(int(channels) // int(reduction), 8)

        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.Hardswish(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_descriptor = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_descriptor = F.adaptive_max_pool2d(x, 1).flatten(1)

        channel_weights = torch.sigmoid(
            self.channel_mlp(avg_descriptor)
            + self.channel_mlp(max_descriptor)
        ).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_weights

        spatial_avg = x.mean(dim=1, keepdim=True)
        spatial_max = x.amax(dim=1, keepdim=True)
        spatial_weights = torch.sigmoid(
            self.spatial_conv(torch.cat((spatial_avg, spatial_max), dim=1))
        )
        return x * spatial_weights


class MultiScaleAttention(nn.Module):
    """Sample-wise softmax attention over the projected feature scales."""

    def __init__(self, channels: int, num_scales: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(channels, 32),
            nn.Hardswish(inplace=True),
            nn.Linear(32, 1, bias=False),
        )
        self.scale_bias = nn.Parameter(torch.zeros(num_scales))

    def forward(self, projected_scales: Sequence[torch.Tensor]) -> torch.Tensor:
        descriptors = [
            F.adaptive_avg_pool2d(feature, 1).flatten(1)
            for feature in projected_scales
        ]
        scores = torch.stack(
            [self.scorer(descriptor).squeeze(1) for descriptor in descriptors],
            dim=1,
        )
        scores = scores + self.scale_bias.unsqueeze(0)
        return torch.softmax(scores, dim=1)


class ScaleProjection(nn.Module):
    """Align one MobileNet feature stage to the common MSAF feature space."""

    def __init__(
        self,
        in_channels: int,
        projection_channels: int = 96,
        attention_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                projection_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(projection_channels),
            nn.Hardswish(inplace=True),
        )
        self.attention = ChannelSpatialAttention(
            projection_channels,
            reduction=attention_reduction,
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        height, width = feature_map.shape[-2:]
        target_h, target_w = output_size

        if height >= target_h and width >= target_w:
            feature_map = F.adaptive_avg_pool2d(
                feature_map, output_size
            )
        else:
            feature_map = F.interpolate(
                feature_map,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return self.attention(self.projection(feature_map))


class MobNetFine(nn.Module):
    """
    MobileNetV3-Large + MSAF multi-scale attention architecture.

    Feature stages:
        features[2]  -> 24 channels
        features[4]  -> 40 channels
        features[7]  -> 80 channels
        features[13] -> 160 channels
        features[16] -> 960 channels

    Progressive fine-tuning:
        classifier_only : train all newly-added MSAF/head layers
        progressive     : additionally unfreeze features[13:]
        full            : unfreeze the complete MobileNetV3 feature extractor
    """

    FEATURE_STAGE_INDICES: Tuple[int, ...] = (2, 4, 7, 13, 16)
    FEATURE_STAGE_CHANNELS: Tuple[int, ...] = (24, 40, 80, 160, 960)

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        projection_channels: int = 96,
        fused_channels: int = 192,
        embedding_dim: int = 384,
        fusion_resolution: int = 14,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")
        if projection_channels < 32:
            raise ValueError("projection_channels must be at least 32.")
        if fused_channels < 64:
            raise ValueError("fused_channels must be at least 64.")
        if embedding_dim < 128:
            raise ValueError("embedding_dim must be at least 128.")
        if fusion_resolution < 7:
            raise ValueError("fusion_resolution must be at least 7.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)

        # The ImageNet classifier is not used by MSAF.
        self.backbone.classifier = nn.Identity()

        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.freeze_backbone = bool(freeze_backbone)
        self.projection_channels = int(projection_channels)
        self.fused_channels = int(fused_channels)
        self.embedding_dim = int(embedding_dim)
        self.fusion_resolution = int(fusion_resolution)
        self.dropout_rate = float(dropout)

        self.scale_branches = nn.ModuleList(
            [
                ScaleProjection(
                    in_channels=channels,
                    projection_channels=self.projection_channels,
                )
                for channels in self.FEATURE_STAGE_CHANNELS
            ]
        )

        self.scale_attention = MultiScaleAttention(
            channels=self.projection_channels,
            num_scales=len(self.FEATURE_STAGE_INDICES),
        )

        concatenated_channels = (
            self.projection_channels
            * len(self.FEATURE_STAGE_INDICES)
        )
        self.fusion_block = nn.Sequential(
            nn.Conv2d(
                concatenated_channels,
                self.fused_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.fused_channels),
            nn.Hardswish(inplace=True),
            ChannelSpatialAttention(self.fused_channels),
        )

        # 1x1 global average + 1x1 global max + 2x2 regional average.
        pooled_dimension = self.fused_channels * 6
        self.embedding = nn.Sequential(
            nn.LayerNorm(pooled_dimension),
            nn.Linear(pooled_dimension, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.LayerNorm(self.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(min(self.dropout_rate, 0.25)),
            nn.Linear(self.embedding_dim // 2, self.num_classes),
        )

        self._initialize_new_layers()

        if self.freeze_backbone:
            for parameter in self.backbone.features.parameters():
                parameter.requires_grad = False

    def _initialize_new_layers(self) -> None:
        # Only initialize newly-added MSAF/head modules; preserve ImageNet weights.
        new_modules = (
            self.scale_branches,
            self.scale_attention,
            self.fusion_block,
            self.embedding,
            self.classifier,
        )
        for root in new_modules:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                    if module.weight is not None:
                        nn.init.ones_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Small batches can destabilize pretrained BatchNorm running statistics.
            # Keep backbone BN statistics fixed while still allowing gradients
            # through any unfrozen backbone convolutional weights.
            for module in self.backbone.features.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def _extract_pyramid(self, images: torch.Tensor) -> List[torch.Tensor]:
        captured: List[torch.Tensor] = []
        wanted = set(self.FEATURE_STAGE_INDICES)
        x = images

        for index, layer in enumerate(self.backbone.features):
            x = layer(x)
            if index in wanted:
                captured.append(x)

        if len(captured) != len(self.FEATURE_STAGE_INDICES):
            raise RuntimeError(
                "Unexpected MobileNetV3-Large feature layout. "
                f"Expected stages {self.FEATURE_STAGE_INDICES}, "
                f"captured {len(captured)} feature maps."
            )

        for index, (feature, expected_channels) in enumerate(
            zip(captured, self.FEATURE_STAGE_CHANNELS)
        ):
            if int(feature.shape[1]) != expected_channels:
                raise RuntimeError(
                    f"MobileNet feature stage {self.FEATURE_STAGE_INDICES[index]} "
                    f"has {feature.shape[1]} channels; expected {expected_channels}. "
                    "Check the torchvision MobileNetV3-Large implementation."
                )

        return captured

    def extract_multiscale_features(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ):
        feature_maps = self._extract_pyramid(images)
        output_size = (
            self.fusion_resolution,
            self.fusion_resolution,
        )

        projected = [
            branch(feature, output_size)
            for branch, feature in zip(self.scale_branches, feature_maps)
        ]

        scale_weights = self.scale_attention(projected)

        weighted_scales = [
            feature * scale_weights[:, index].view(-1, 1, 1, 1)
            for index, feature in enumerate(projected)
        ]
        concatenated = torch.cat(weighted_scales, dim=1)
        fused_map = self.fusion_block(concatenated)

        if return_attention:
            return fused_map, scale_weights
        return fused_map

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        fused_map = self.extract_multiscale_features(images)
        global_average = F.adaptive_avg_pool2d(fused_map, 1).flatten(1)
        global_maximum = F.adaptive_max_pool2d(fused_map, 1).flatten(1)
        regional_average = F.adaptive_avg_pool2d(
            fused_map, (2, 2)
        ).flatten(1)
        pooled = torch.cat(
            (global_average, global_maximum, regional_average),
            dim=1,
        )
        return self.embedding(pooled)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                "Expected input shape [B, C, H, W]. "
                f"Received {tuple(images.shape)}."
            )
        if images.shape[1] != 3:
            raise ValueError(
                f"Expected 3 input channels, received {images.shape[1]}."
            )
        if images.shape[-2] < 32 or images.shape[-1] < 32:
            raise ValueError("Input height and width must be at least 32.")

        embeddings = self.forward_features(images)
        return self.classifier(embeddings)

    def get_target_layer(self) -> nn.Module:
        """Spatial convolutional feature layer used by Grad-CAM / Grad-CAM++."""
        return self.backbone.features[-1]

    def get_head_modules(self) -> Tuple[nn.Module, ...]:
        """All newly-added modules trained during the classifier-only stage."""
        return (
            self.scale_branches,
            self.scale_attention,
            self.fusion_block,
            self.embedding,
            self.classifier,
        )

    def get_architecture_config(self) -> Dict[str, object]:
        return {
            "model_name": "MobNetFine",
            "architecture": "MobileNetV3-Large Multi-Scale Attention Feature Fusion",
            "architecture_version": "1.0",
            "starting_weights": "ImageNet pretrained" if self.pretrained else "Random initialization",
            "backbone": "MobileNetV3-Large",
            "feature_stage_indices": list(self.FEATURE_STAGE_INDICES),
            "feature_stage_channels": list(self.FEATURE_STAGE_CHANNELS),
            "scale_projection_channels": self.projection_channels,
            "fusion_resolution": self.fusion_resolution,
            "fused_channels": self.fused_channels,
            "scale_attention": "Shared MLP scoring with sample-wise softmax across five feature scales",
            "feature_attention": "CBAM-style channel and spatial attention in each scale and fused map",
            "fusion_method": "Attention-weighted multi-scale concatenation followed by 1x1 convolution",
            "pooling": [
                "global_average",
                "global_max",
                "2x2_regional_average",
            ],
            "embedding_dimension": self.embedding_dim,
            "classifier_dropout": self.dropout_rate,
            "num_classes": self.num_classes,
            "output_activation": "None (raw logits)",
            "gradcam_target_layer": "backbone.features[-1]",
            "backbone_batchnorm_statistics": "Frozen during training",
        }


def build_mobnetfine(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **model_kwargs,
) -> MobNetFine:
    """Factory-compatible model builder."""
    return MobNetFine(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        **model_kwargs,
    )


def set_trainable_stage(model: MobNetFine, stage: str) -> Dict[str, int]:
    """Configure classifier-only, progressive-tail, or full fine-tuning."""
    if stage not in {"classifier_only", "progressive", "full"}:
        raise ValueError(f"Unknown MobNetFine training stage: {stage}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.get_head_modules():
        for parameter in module.parameters():
            parameter.requires_grad = True

    if stage == "progressive":
        for parameter in model.backbone.features[13:].parameters():
            parameter.requires_grad = True
    elif stage == "full":
        for parameter in model.backbone.features.parameters():
            parameter.requires_grad = True

    model.active_training_stage = stage

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable <= 0:
        raise RuntimeError(f"MobNetFine stage '{stage}' has no trainable parameters.")

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def _unique_trainable(modules: Iterable[nn.Module]) -> List[nn.Parameter]:
    parameters: List[nn.Parameter] = []
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def get_stage_parameter_groups(
    model: MobNetFine,
    stage: str,
    rates: Dict[str, float],
    weight_decay: float,
):
    """Create differential-LR parameter groups with complete parameter coverage."""
    if stage == "classifier_only":
        specifications = [
            ("msaf_head", list(model.get_head_modules()), rates["head"]),
        ]
    elif stage == "progressive":
        specifications = [
            ("mobilenet_tail", [model.backbone.features[13:]], rates["backbone"]),
            ("msaf_head", list(model.get_head_modules()), rates["head"]),
        ]
    elif stage == "full":
        specifications = [
            ("mobilenet_backbone", [model.backbone.features], rates["backbone"]),
            ("msaf_head", list(model.get_head_modules()), rates["head"]),
        ]
    else:
        raise ValueError(f"Unknown MobNetFine training stage: {stage}")

    groups = []
    for name, modules, learning_rate in specifications:
        parameters = _unique_trainable(modules)
        if parameters:
            groups.append(
                {
                    "name": name,
                    "params": parameters,
                    "lr": float(learning_rate),
                    "weight_decay": float(weight_decay),
                }
            )

    covered = {
        id(parameter)
        for group in groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if covered != expected:
        raise RuntimeError(
            "MobNetFine optimizer parameter coverage error: "
            f"missing={len(expected - covered)}, "
            f"extra={len(covered - expected)}."
        )
    return groups


def get_parameter_information(model: nn.Module) -> Dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen = total - trainable
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_percentage": 100.0 * trainable / total if total else 0.0,
    }


def main() -> None:
    """CPU smoke test for the model, gradient flow, and progressive stages."""
    print("=" * 72)
    print("MOBNETFINE - MSAF ARCHITECTURE TEST")
    print("=" * 72)

    model = build_mobnetfine(
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False,
    )
    model.train()

    images = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, NUM_CLASSES, (2,))
    logits = model(images)

    assert logits.shape == (2, NUM_CLASSES)
    assert torch.isfinite(logits).all()
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("[PASS] Forward and backward propagation")

    target_layer = model.get_target_layer()
    assert isinstance(target_layer, nn.Module)
    print("[PASS] Grad-CAM target layer")

    for stage in ("classifier_only", "progressive", "full"):
        info = set_trainable_stage(model, stage)
        groups = get_stage_parameter_groups(
            model,
            stage,
            rates={"backbone": 1e-5, "head": 1e-4},
            weight_decay=1e-4,
        )
        optimized = sum(
            parameter.numel()
            for group in groups
            for parameter in group["params"]
        )
        assert optimized == info["trainable_parameters"]
        print(
            f"[PASS] {stage}: "
            f"{info['trainable_parameters']:,} trainable parameters"
        )

    print("\nMOBNETFINE ARCHITECTURE TEST: PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
