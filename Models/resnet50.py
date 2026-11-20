
"""
Maize Leaf Classification
ResNet50 Architecture

Features:
    - ImageNet pretrained weights.
    - Training from scratch option.
    - Backbone freezing/unfreezing.
    - Seven-class classification head.
    - Feature extraction.
    - Grad-CAM target layer access.
    - Parameter information.
    - Architecture configuration.

Input:
    [B, 3, 224, 224]

Output:
    [B, 7] raw logits.
"""

import torch
import torch.nn as nn
import torchvision

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

from src.data.class_names import NUM_CLASSES


# ============================================================
# RESNET50 MODEL
# ============================================================

class ResNet50Classifier(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.0
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
        # Load pretrained or randomly initialized ResNet50
        # ----------------------------------------------------

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet50(
            weights=weights
        )

        # ----------------------------------------------------
        # Get original classifier input features
        # ----------------------------------------------------

        in_features = self.backbone.fc.in_features

        # ----------------------------------------------------
        # Replace original ImageNet classifier
        # ----------------------------------------------------

        if dropout > 0:

            self.backbone.fc = nn.Sequential(

                nn.Dropout(p=dropout),

                nn.Linear(
                    in_features,
                    num_classes
                )

            )

        else:

            self.backbone.fc = nn.Linear(
                in_features,
                num_classes
            )

        # ----------------------------------------------------
        # Freeze backbone if requested
        # ----------------------------------------------------

        if freeze_backbone:

            for name, parameter in (
                self.backbone.named_parameters()
            ):

                parameter.requires_grad = (
                    name.startswith("fc.")
                )

    # ========================================================
    # CONTROL BATCH NORMALIZATION WHEN FROZEN
    # ========================================================

    def train(self, mode=True):

        super().train(mode)

        if self.freeze_backbone:

            # Keep frozen backbone BatchNorm statistics fixed.

            for name, module in self.backbone.named_modules():

                if isinstance(module, nn.BatchNorm2d):

                    module.eval()

        return self

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def forward_features(self, x):

        model = self.backbone

        x = model.conv1(x)

        x = model.bn1(x)

        x = model.relu(x)

        x = model.maxpool(x)

        x = model.layer1(x)

        x = model.layer2(x)

        x = model.layer3(x)

        x = model.layer4(x)

        x = model.avgpool(x)

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

        return self.backbone.layer4[-1]

    # ========================================================
    # ARCHITECTURE CONFIGURATION
    # ========================================================

    def get_architecture_config(self):

        return {

            "model_name": "ResNet50",

            "architecture_type": "Residual CNN",

            "depth": 50,

            "block_type": "Bottleneck",

            "block_configuration": [3, 4, 6, 3],

            "num_classes": self.num_classes,

            "input_channels": 3,

            "pretrained": self.pretrained,

            "pretrained_source": (
                "ImageNet"
                if self.pretrained
                else "Random initialization"
            ),

            "freeze_backbone": self.freeze_backbone,

            "feature_dimensions": 2048,

            "classifier": str(self.backbone.fc),

            "dropout": self.dropout_rate,

            "global_pooling": "Adaptive Average Pooling",

            "output_activation": "None (raw logits)",

            "gradcam_target_layer": (
                "backbone.layer4[-1]"
            ),

            "torchvision_version": torchvision.__version__,

        }


# ============================================================
# MODEL BUILDER
# ============================================================

def build_resnet50(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=False,
    **kwargs
):

    return ResNet50Classifier(

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

    print("MAIZE LEAF - RESNET50 ARCHITECTURE TEST")

    print("=" * 70)

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
    # Build model WITHOUT downloading pretrained weights
    # --------------------------------------------------------

    print("\nBuilding ResNet50 from scratch...")

    model = build_resnet50(
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False
    )

    model = model.to(device)

    print("[PASS] Model initialization")

    # --------------------------------------------------------
    # Architecture information
    # --------------------------------------------------------

    architecture = model.get_architecture_config()

    assert architecture["depth"] == 50

    assert architecture["num_classes"] == 7

    assert architecture["block_configuration"] == [
        3, 4, 6, 3
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
        +
        parameter_info["frozen_parameters"]
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

    assert features.shape == (2, 2048)

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

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    print("[PASS] Backpropagation")

    # --------------------------------------------------------
    # Grad-CAM target layer
    # --------------------------------------------------------

    target_layer = model.get_target_layer()

    assert isinstance(target_layer, nn.Module)

    print(
        f"[PASS] Grad-CAM target layer: "
        f"{target_layer.__class__.__name__}"
    )

    # --------------------------------------------------------
    # Test frozen backbone
    # --------------------------------------------------------

    print("\nTesting frozen backbone...")

    del model

    if device.type == "cuda":

        torch.cuda.empty_cache()

    frozen_model = build_resnet50(
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=True
    ).to(device)

    frozen_info = get_parameter_information(
        frozen_model
    )

    assert frozen_info["frozen_parameters"] > 0

    assert (
        frozen_info["trainable_parameters"]
        ==
        sum(
            p.numel()
            for p in frozen_model.backbone.fc.parameters()
        )
    )

    frozen_model.train()

    for name, module in frozen_model.backbone.named_modules():

        if isinstance(module, nn.BatchNorm2d):

            assert not module.training

    print("[PASS] Backbone parameter freezing")

    print("[PASS] Frozen BatchNorm running statistics")

    # --------------------------------------------------------
    # Test unfrozen classifier
    # --------------------------------------------------------

    assert all(
        p.requires_grad
        for p in frozen_model.backbone.fc.parameters()
    )

    print("[PASS] Classifier remains trainable")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    del frozen_model

    if device.type == "cuda":

        torch.cuda.empty_cache()

    print("\nRESNET50 ARCHITECTURE TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
