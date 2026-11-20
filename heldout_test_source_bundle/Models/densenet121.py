
"""
Maize Leaf Classification
DenseNet121 Architecture

Features:
    - ImageNet pretrained weights.
    - Training from scratch.
    - Backbone freezing/unfreezing.
    - Seven-class classification head.
    - Dense connectivity.
    - Feature reuse.
    - Feature extraction.
    - Grad-CAM target layer access.
    - Parameter information.

Input:
    [B, 3, 224, 224]

Output:
    [B, 7] raw logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models import (
    densenet121,
    DenseNet121_Weights,
)

from src.data.class_names import NUM_CLASSES


# ============================================================
# DENSENET121 MODEL
# ============================================================

class DenseNet121Classifier(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.2
    ):

        super().__init__()

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if num_classes <= 1:

            raise ValueError(
                "Number of classes must be greater than 1."
            )

        if not 0 <= dropout < 1:

            raise ValueError(
                "Dropout must be in [0, 1)."
            )

        self.num_classes = num_classes

        self.pretrained = pretrained

        self.freeze_backbone = freeze_backbone

        self.dropout_rate = dropout

        # ----------------------------------------------------
        # Load pretrained or randomly initialized DenseNet121
        # ----------------------------------------------------

        weights = (
            DenseNet121_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = densenet121(
            weights=weights
        )

        # ----------------------------------------------------
        # Replace original ImageNet classifier
        # ----------------------------------------------------

        in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(

            nn.Dropout(p=dropout),

            nn.Linear(
                in_features,
                num_classes
            )

        )

        # ----------------------------------------------------
        # Freeze feature extraction backbone if requested
        # ----------------------------------------------------

        if freeze_backbone:

            for parameter in self.backbone.features.parameters():

                parameter.requires_grad = False

    # ========================================================
    # CONTROL BATCH NORMALIZATION WHEN FROZEN
    # ========================================================

    def train(self, mode=True):

        super().train(mode)

        if self.freeze_backbone:

            for module in self.backbone.features.modules():

                if isinstance(module, nn.BatchNorm2d):

                    module.eval()

        return self

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def forward_features(self, x):

        # DenseNet feature extraction.

        x = self.backbone.features(x)

        # Final ReLU activation.

        x = F.relu(
            x,
            inplace=False
        )

        # Global average pooling.

        x = F.adaptive_avg_pool2d(
            x,
            output_size=(1, 1)
        )

        # Flatten feature maps.

        x = torch.flatten(x, 1)

        return x

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def forward(self, x):

        if x.ndim != 4:

            raise ValueError(
                "Expected input shape [B, C, H, W]. "
                f"Received: {tuple(x.shape)}"
            )

        if x.shape[1] != 3:

            raise ValueError(
                f"Expected 3 input channels, got {x.shape[1]}."
            )

        if x.shape[2] < 32 or x.shape[3] < 32:

            raise ValueError(
                "Input height and width must be at least 32."
            )

        return self.backbone(x)

    # ========================================================
    # GRAD-CAM TARGET LAYER
    # ========================================================

    def get_target_layer(self):

        return (
            self.backbone
            .features
            .denseblock4
            .denselayer16
            .conv2
        )

    # ========================================================
    # ARCHITECTURE CONFIGURATION
    # ========================================================

    def get_architecture_config(self):

        return {

            "model_name": "DenseNet121",

            "architecture_type": "Dense Convolutional Network",

            "depth": 121,

            "dense_blocks": 4,

            "block_configuration": [6, 12, 24, 16],

            "connectivity": "Dense connectivity",

            "num_classes": self.num_classes,

            "input_channels": 3,

            "pretrained": self.pretrained,

            "pretrained_source": (
                "ImageNet"
                if self.pretrained
                else "Random initialization"
            ),

            "freeze_backbone": self.freeze_backbone,

            "feature_dimensions": 1024,

            "classifier": str(
                self.backbone.classifier
            ),

            "dropout": self.dropout_rate,

            "global_pooling": "Adaptive Average Pooling",

            "output_activation": "None (raw logits)",

            "gradcam_target_layer": (
                "backbone.features.denseblock4."
                "denselayer16.conv2"
            ),

            "torchvision_version": torchvision.__version__,

        }


# ============================================================
# MODEL BUILDER
# ============================================================

def build_densenet121(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=False,
    **kwargs
):

    return DenseNet121Classifier(

        num_classes=num_classes,

        pretrained=pretrained,

        freeze_backbone=freeze_backbone,

        **kwargs

    )


# ============================================================
# PARAMETER INFORMATION
# ============================================================

def get_parameter_information(model):

    total_parameters = sum(

        parameter.numel()

        for parameter in model.parameters()

    )

    trainable_parameters = sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )

    frozen_parameters = (
        total_parameters - trainable_parameters
    )

    return {

        "total_parameters": total_parameters,

        "trainable_parameters": trainable_parameters,

        "frozen_parameters": frozen_parameters,

        "trainable_percentage": (

            100.0 * trainable_parameters / total_parameters

            if total_parameters else 0.0

        ),

    }


# ============================================================
# MODEL TEST
# ============================================================

def main():

    print("=" * 70)

    print("MAIZE LEAF - DENSENET121 ARCHITECTURE TEST")

    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(

        "cuda" if torch.cuda.is_available()
        else "cpu"

    )

    print(f"\nDevice: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Build model from scratch
    # --------------------------------------------------------

    print("\nBuilding DenseNet121 from scratch...")

    model = build_densenet121(

        num_classes=NUM_CLASSES,

        pretrained=False,

        freeze_backbone=False

    ).to(device)

    print("[PASS] Model initialization")

    # --------------------------------------------------------
    # Architecture configuration
    # --------------------------------------------------------

    architecture = model.get_architecture_config()

    assert architecture["num_classes"] == 7

    assert architecture["depth"] == 121

    assert architecture["feature_dimensions"] == 1024

    assert architecture["dense_blocks"] == 4

    assert architecture["block_configuration"] == [
        6, 12, 24, 16
    ]

    print("[PASS] Architecture configuration")

    # --------------------------------------------------------
    # Parameter information
    # --------------------------------------------------------

    parameter_info = get_parameter_information(model)

    print("\nPARAMETER INFORMATION")

    for key, value in parameter_info.items():

        if isinstance(value, float):

            print(f"{key:<30}: {value:.2f}%")

        else:

            print(f"{key:<30}: {value:,}")

    assert parameter_info["total_parameters"] > 0

    assert (

        parameter_info["total_parameters"]

        ==

        parameter_info["trainable_parameters"]

        + parameter_info["frozen_parameters"]

    )

    print("[PASS] Parameter count")

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.train()

    inputs = torch.randn(

        2, 3, 224, 224,

        device=device

    )

    targets = torch.randint(

        0, NUM_CLASSES, (2,),

        device=device

    )

    outputs = model(inputs)

    assert outputs.shape == (2, NUM_CLASSES)

    assert torch.isfinite(outputs).all()

    print("\n[PASS] Forward propagation")

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    features = model.forward_features(inputs)

    assert features.shape == (2, 1024)

    assert torch.isfinite(features).all()

    print(
        f"[PASS] Feature extraction: {tuple(features.shape)}"
    )

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    loss = criterion(outputs, targets)

    loss.backward()

    assert torch.isfinite(loss)

    gradient_parameters = [

        parameter

        for parameter in model.parameters()

        if parameter.grad is not None

    ]

    assert len(gradient_parameters) > 0

    for parameter in gradient_parameters:

        assert torch.isfinite(parameter.grad).all()

    print("[PASS] Backpropagation")

    # --------------------------------------------------------
    # Grad-CAM target layer
    # --------------------------------------------------------

    target_layer = model.get_target_layer()

    assert isinstance(target_layer, nn.Conv2d)

    print(
        f"[PASS] Grad-CAM target layer: "
        f"{target_layer.__class__.__name__}"
    )

    # --------------------------------------------------------
    # Release first model
    # --------------------------------------------------------

    del model

    if device.type == "cuda":

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Test frozen backbone
    # --------------------------------------------------------

    print("\nTesting frozen backbone...")

    frozen_model = build_densenet121(

        num_classes=NUM_CLASSES,

        pretrained=False,

        freeze_backbone=True

    ).to(device)

    frozen_info = get_parameter_information(
        frozen_model
    )

    assert frozen_info["frozen_parameters"] > 0

    classifier_parameters = sum(

        parameter.numel()

        for parameter in frozen_model.backbone.classifier.parameters()

    )

    assert (

        frozen_info["trainable_parameters"]

        == classifier_parameters

    )

    print("[PASS] Backbone parameter freezing")

    # --------------------------------------------------------
    # BatchNorm verification
    # --------------------------------------------------------

    frozen_model.train()

    for module in frozen_model.backbone.features.modules():

        if isinstance(module, nn.BatchNorm2d):

            assert not module.training

    print("[PASS] Frozen BatchNorm running statistics")

    # --------------------------------------------------------
    # Classifier verification
    # --------------------------------------------------------

    assert all(

        parameter.requires_grad

        for parameter in frozen_model.backbone.classifier.parameters()

    )

    print("[PASS] Classifier remains trainable")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    del frozen_model

    if device.type == "cuda":

        torch.cuda.empty_cache()

    print("\nDENSENET121 ARCHITECTURE TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
