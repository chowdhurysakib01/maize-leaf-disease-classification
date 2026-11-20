"""
MaizeLeaf-DMNet v2
Dual-backbone MobileNetV3-Large + DenseNet121 with attention-guided,
multi-scale feature fusion.

Compatibility:
- Keeps the existing model key: maizeleaf_dmnet
- Keeps the existing builder: build_maizeleaf_dmnet
- Returns raw seven-class logits for the existing trainer
- Exposes get_target_layer() for Grad-CAM / Grad-CAM++
- Exposes get_architecture_config() for existing metadata export

Important: use --pretrained in the trainer command. The previous run used
--no-pretrained, so both backbones were trained from random initialization.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    DenseNet121_Weights,
    MobileNet_V3_Large_Weights,
    densenet121,
    mobilenet_v3_large,
)


class CBAM(nn.Module):
    """Lightweight channel and spatial attention for a feature map."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_descriptor = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_descriptor = F.adaptive_max_pool2d(x, 1).flatten(1)

        channel_attention = torch.sigmoid(
            self.channel_mlp(avg_descriptor)
            + self.channel_mlp(max_descriptor)
        ).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_attention

        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max = torch.amax(x, dim=1, keepdim=True)
        spatial_attention = torch.sigmoid(
            self.spatial_conv(torch.cat((spatial_avg, spatial_max), dim=1))
        )
        return x * spatial_attention


class MultiScaleBranchEncoder(nn.Module):
    """Projects a backbone map and encodes global plus regional evidence."""

    def __init__(
        self,
        in_channels: int,
        projection_channels: int = 256,
        embedding_dim: int = 512,
        dropout: float = 0.20,
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
            CBAM(projection_channels),
        )

        # 1x1 average + 1x1 max + 2x2 regional average = 6 vectors.
        pooled_dim = projection_channels * 6
        self.embedding = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        feature_map = self.projection(feature_map)

        global_avg = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        global_max = F.adaptive_max_pool2d(feature_map, 1).flatten(1)
        regional_avg = F.adaptive_avg_pool2d(feature_map, (2, 2)).flatten(1)

        multi_scale = torch.cat(
            (global_avg, global_max, regional_avg), dim=1
        )
        return self.embedding(multi_scale)


class MaizeLeafDMNet(nn.Module):
    """Attention-guided dual-backbone classifier."""

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.35,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if hidden_dim < 128:
            raise ValueError("hidden_dim must be at least 128.")

        mobile_weights = (
            MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )
        dense_weights = DenseNet121_Weights.DEFAULT if pretrained else None

        mobile_model = mobilenet_v3_large(weights=mobile_weights)
        dense_model = densenet121(weights=dense_weights)

        self.mobilenet_features = mobile_model.features
        self.densenet_features = dense_model.features
        del mobile_model, dense_model

        self.mobilenet_feature_dim = 960
        self.densenet_feature_dim = 1024
        self.branch_embedding_dim = int(hidden_dim)

        self.mobile_encoder = MultiScaleBranchEncoder(
            in_channels=self.mobilenet_feature_dim,
            projection_channels=256,
            embedding_dim=self.branch_embedding_dim,
            dropout=0.20,
        )
        self.dense_encoder = MultiScaleBranchEncoder(
            in_channels=self.densenet_feature_dim,
            projection_channels=256,
            embedding_dim=self.branch_embedding_dim,
            dropout=0.20,
        )

        fusion_dim = self.branch_embedding_dim * 4
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.branch_embedding_dim * 2, self.branch_embedding_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=min(dropout, 0.30)),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.freeze_backbone = bool(freeze_backbone)
        self.dropout_rate = float(dropout)
        self.hidden_dim = int(hidden_dim)
        self.fusion_dim = int(fusion_dim)

        if self.freeze_backbone:
            self._set_backbone_gradients(False)

        self._initialize_new_layers()

    def _set_backbone_gradients(self, trainable: bool) -> None:
        for module in (self.mobilenet_features, self.densenet_features):
            for parameter in module.parameters():
                parameter.requires_grad = trainable

    def _initialize_new_layers(self) -> None:
        # Do not reinitialize the ImageNet-pretrained backbone weights.
        new_modules = (
            self.mobile_encoder,
            self.dense_encoder,
            self.fusion_gate,
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
        if mode and self.freeze_backbone:
            self.mobilenet_features.eval()
            self.densenet_features.eval()
        return self

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        mobile_map = self.mobilenet_features(images)
        dense_map = self.densenet_features(images)
        dense_map = F.relu(dense_map, inplace=False)

        mobile_embedding = self.mobile_encoder(mobile_map)
        dense_embedding = self.dense_encoder(dense_map)

        joined = torch.cat((mobile_embedding, dense_embedding), dim=1)
        gate = self.fusion_gate(joined)
        gated_fusion = (
            gate * mobile_embedding
            + (1.0 - gate) * dense_embedding
        )
        difference = torch.abs(mobile_embedding - dense_embedding)

        # Keep branch-specific evidence while adding adaptive cross-branch fusion.
        return torch.cat(
            (mobile_embedding, dense_embedding, gated_fusion, difference),
            dim=1,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        fused_features = self.extract_features(images)
        return self.classifier(fused_features)

    def get_target_layer(self) -> nn.Module:
        """Spatial convolution layer used by Grad-CAM / Grad-CAM++."""
        return (
            self.densenet_features
            .denseblock4
            .denselayer16
            .conv2
        )

    def get_architecture_config(self) -> Dict[str, object]:
        return {
            "architecture": "attention-guided multi-scale dual-backbone fusion",
            "model_name": "MaizeLeaf-DMNet",
            "architecture_version": "2.0",
            "mobilenet_backbone": "MobileNetV3-Large",
            "densenet_backbone": "DenseNet121",
            "mobilenet_feature_dim": self.mobilenet_feature_dim,
            "densenet_feature_dim": self.densenet_feature_dim,
            "branch_projection_channels": 256,
            "branch_embedding_dim": self.branch_embedding_dim,
            "attention": "CBAM channel and spatial attention",
            "pooling": ["global_average", "global_max", "2x2_regional_average"],
            "fusion_method": "learned_gated_fusion_plus_branch_residuals_and_absolute_difference",
            "fusion_feature_dim": self.fusion_dim,
            "classifier_hidden_dim": self.hidden_dim,
            "classifier_dropout": self.dropout_rate,
            "pretrained_backbones": self.pretrained,
            "backbone_frozen": self.freeze_backbone,
            "num_classes": self.num_classes,
        }


def build_maizeleaf_dmnet(
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **model_kwargs,
) -> MaizeLeafDMNet:
    """Factory-compatible builder used by Models.model_factory."""
    return MaizeLeafDMNet(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        **model_kwargs,
    )
