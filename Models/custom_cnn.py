
"""
Maize Leaf Classification
Custom CNN Baseline Architecture

Architecture:
    Input (3 x 224 x 224)
        |
    Conv Stage 1: 32 channels
        |
    Conv Stage 2: 64 channels
        |
    Conv Stage 3: 128 channels
        |
    Conv Stage 4: 256 channels
        |
    Global Average Pooling
        |
    Dropout
        |
    Fully Connected Classifier
        |
    Output: 7 classes

Important:
    - Outputs raw logits (no softmax).
    - Supports configurable number of classes.
    - Supports configurable channel width.
    - Designed for training from scratch.
    - No pretrained weights.
"""

import torch
import torch.nn as nn

from src.data.class_names import NUM_CLASSES


# ============================================================
# CONVOLUTIONAL STAGE
# ============================================================

class ConvStage(nn.Module):
    """
    Two convolutional layers followed by max pooling.

    Structure:
        Conv2D
        BatchNorm
        ReLU
        Conv2D
        BatchNorm
        ReLU
        MaxPool
        Dropout2D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_dropout=0.05
    ):

        super().__init__()

        if not 0 <= spatial_dropout < 1:

            raise ValueError(
                "Spatial dropout must be in [0, 1)."
            )

        layers = [

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

        ]

        if spatial_dropout > 0:

            layers.append(
                nn.Dropout2d(
                    p=spatial_dropout
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)


# ============================================================
# CUSTOM CNN
# ============================================================

class CustomCNN(nn.Module):
    """
    Custom convolutional neural network for
    seven-class maize leaf classification.

    Default architecture:

        3 -> 32 -> 64 -> 128 -> 256

    Four convolutional stages.

    Input:
        [B, 3, H, W]

    Output:
        [B, num_classes]

    Output values are raw logits.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        in_channels=3,
        base_channels=32,
        dropout=0.35,
        spatial_dropout=0.05
    ):

        super().__init__()

        # ----------------------------------------------------
        # Validate arguments
        # ----------------------------------------------------

        if num_classes <= 1:

            raise ValueError(
                "Number of classes must be greater than 1."
            )

        if in_channels <= 0:

            raise ValueError(
                "Input channels must be positive."
            )

        if base_channels <= 0:

            raise ValueError(
                "Base channels must be positive."
            )

        if not 0 <= dropout < 1:

            raise ValueError(
                "Classifier dropout must be in [0, 1)."
            )

        if not 0 <= spatial_dropout < 1:

            raise ValueError(
                "Spatial dropout must be in [0, 1)."
            )

        self.num_classes = num_classes

        self.in_channels = in_channels

        self.base_channels = base_channels

        self.dropout_rate = dropout

        self.spatial_dropout = spatial_dropout

        # ----------------------------------------------------
        # Channel configuration
        # ----------------------------------------------------

        channels = [

            base_channels,

            base_channels * 2,

            base_channels * 4,

            base_channels * 8,

        ]

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        self.features = nn.Sequential(

            ConvStage(
                in_channels,
                channels[0],
                spatial_dropout
            ),

            ConvStage(
                channels[0],
                channels[1],
                spatial_dropout
            ),

            ConvStage(
                channels[1],
                channels[2],
                spatial_dropout
            ),

            ConvStage(
                channels[2],
                channels[3],
                spatial_dropout
            ),

        )

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(p=dropout),

            nn.Linear(
                channels[-1],
                num_classes
            ),

        )

        # ----------------------------------------------------
        # Initialize parameters
        # ----------------------------------------------------

        self._initialize_weights()

    # ========================================================
    # WEIGHT INITIALIZATION
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if module.bias is not None:

                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):

                nn.init.ones_(module.weight)

                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.zeros_(module.bias)

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def forward_features(self, x):

        return self.features(x)

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def forward(self, x):

        if x.ndim != 4:

            raise ValueError(
                "Expected input shape [B, C, H, W]. "
                f"Received: {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:

            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"received {x.shape[1]}."
            )

        if x.shape[2] < 16 or x.shape[3] < 16:

            raise ValueError(
                "Input height and width must be at least 16."
            )

        # Extract features.

        x = self.forward_features(x)

        # Global average pooling.

        x = self.global_pool(x)

        # Classification.

        logits = self.classifier(x)

        return logits

    # ========================================================
    # ARCHITECTURE CONFIGURATION
    # ========================================================

    def get_architecture_config(self):

        return {

            "model_name": "CustomCNN",

            "architecture_type": "Custom CNN",

            "input_channels": self.in_channels,

            "num_classes": self.num_classes,

            "base_channels": self.base_channels,

            "stage_channels": [

                self.base_channels,

                self.base_channels * 2,

                self.base_channels * 4,

                self.base_channels * 8,

            ],

            "number_of_stages": 4,

            "convolutions_per_stage": 2,

            "global_pooling": "AdaptiveAvgPool2d(1)",

            "classifier_dropout": self.dropout_rate,

            "spatial_dropout": self.spatial_dropout,

            "pretrained": False,

            "output_activation": "None (raw logits)",

        }


# ============================================================
# MODEL BUILDER
# ============================================================

def build_custom_cnn(
    num_classes=NUM_CLASSES,
    **kwargs
):
    """
    Standardized model creation function.

    This function will later be called by model_factory.py.
    """

    return CustomCNN(
        num_classes=num_classes,
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

    }


# ============================================================
# MODEL TEST
# ============================================================

def main():

    print("=" * 70)

    print("MAIZE LEAF - CUSTOM CNN ARCHITECTURE TEST")

    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(

        "cuda" if torch.cuda.is_available()
        else "cpu"

    )

    print(f"\nSelected device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_custom_cnn(
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    print("\n[PASS] Model initialization")

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    print("\nMODEL ARCHITECTURE")

    print(model)

    architecture = model.get_architecture_config()

    assert architecture["num_classes"] == 7

    assert architecture["number_of_stages"] == 4

    print("\n[PASS] Architecture configuration")

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameter_info = get_parameter_information(model)

    print("\nPARAMETER INFORMATION")

    for key, value in parameter_info.items():

        print(f"{key:<30}: {value:,}")

    assert parameter_info["total_parameters"] > 0

    assert (
        parameter_info["total_parameters"]
        ==
        parameter_info["trainable_parameters"]
        +
        parameter_info["frozen_parameters"]
    )

    print("\n[PASS] Parameter count")

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.train()

    inputs = torch.randn(
        2,
        3,
        224,
        224,
        device=device
    )

    targets = torch.randint(
        0,
        NUM_CLASSES,
        (2,),
        device=device
    )

    outputs = model(inputs)

    print("\nFORWARD PASS")

    print(f"Input shape : {tuple(inputs.shape)}")

    print(f"Output shape: {tuple(outputs.shape)}")

    assert outputs.shape == (2, NUM_CLASSES)

    assert torch.isfinite(outputs).all()

    print("[PASS] Forward propagation")

    # --------------------------------------------------------
    # Loss and backward pass
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        outputs,
        targets
    )

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

    print("\nBACKWARD PASS")

    print(f"Test loss: {loss.item():.6f}")

    print(
        f"Parameters receiving gradients: "
        f"{len(gradient_parameters)}"
    )

    print("[PASS] Backpropagation")

    # --------------------------------------------------------
    # Output verification
    # --------------------------------------------------------

    assert outputs.shape[1] == NUM_CLASSES

    print("\n[PASS] Seven-class output")

    # --------------------------------------------------------
    # State dictionary
    # --------------------------------------------------------

    state = model.state_dict()

    assert len(state) > 0

    print("[PASS] State dictionary generation")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    del model

    if device.type == "cuda":

        torch.cuda.empty_cache()

    print("\nCUSTOM CNN ARCHITECTURE TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
