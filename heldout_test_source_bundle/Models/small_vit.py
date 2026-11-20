
"""
Maize Leaf Classification
Small Vision Transformer (Small ViT)

Custom lightweight Vision Transformer.

Architecture:
    Image: 224 x 224
    Patch: 16 x 16
    Number of patches: 196
    Embedding dimension: 192
    Transformer depth: 6
    Attention heads: 3
    MLP ratio: 4
    Classes: 7

Features:
    - Custom ViT implementation.
    - Training from scratch.
    - Backbone freezing/unfreezing.
    - Multi-head self-attention.
    - Learnable positional embeddings.
    - Class token.
    - Feature extraction.
    - Grad-CAM target layer access.
    - Parameter information.

Output:
    [B, 7] raw logits.

Note:
    This custom architecture does not support ImageNet
    pretrained weights. Use --no-pretrained.
"""

import torch
import torch.nn as nn
import torchvision

from src.data.class_names import NUM_CLASSES


# ============================================================
# TRANSFORMER ENCODER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    """
    Pre-normalized Transformer encoder block.

    Structure:

        LayerNorm
            |
        Multi-Head Self Attention
            |
        Residual Connection
            |
        LayerNorm
            |
        MLP (GELU)
            |
        Residual Connection
    """

    def __init__(
        self,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.0
    ):

        super().__init__()

        hidden_dim = int(
            embed_dim * mlp_ratio
        )

        # ----------------------------------------------------
        # First normalization
        # ----------------------------------------------------

        self.norm1 = nn.LayerNorm(
            embed_dim,
            eps=1e-6
        )

        # ----------------------------------------------------
        # Multi-head self-attention
        # ----------------------------------------------------

        self.attn = nn.MultiheadAttention(

            embed_dim=embed_dim,

            num_heads=num_heads,

            dropout=attention_dropout,

            batch_first=True,

        )

        self.attn_dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Second normalization
        # ----------------------------------------------------

        self.norm2 = nn.LayerNorm(
            embed_dim,
            eps=1e-6
        )

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        self.mlp = nn.Sequential(

            nn.Linear(
                embed_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                embed_dim
            ),

            nn.Dropout(dropout),

        )

    def forward(self, x):

        # Multi-head self-attention.

        normalized = self.norm1(x)

        attention_output, _ = self.attn(

            normalized,

            normalized,

            normalized,

            need_weights=False

        )

        # First residual connection.

        x = x + self.attn_dropout(
            attention_output
        )

        # MLP with second residual connection.

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ============================================================
# SMALL VISION TRANSFORMER
# ============================================================

class SmallViT(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.0,
        freeze_backbone=False,
        pretrained=False
    ):

        super().__init__()

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        if num_classes <= 1:

            raise ValueError(
                "Number of classes must be greater than 1."
            )

        if image_size <= 0 or patch_size <= 0:

            raise ValueError(
                "Image size and patch size must be positive."
            )

        if image_size % patch_size != 0:

            raise ValueError(
                "Image size must be divisible by patch size."
            )

        if embed_dim <= 0 or depth <= 0:

            raise ValueError(
                "Embedding dimension and depth must be positive."
            )

        if num_heads <= 0 or embed_dim % num_heads != 0:

            raise ValueError(
                "Embedding dimension must be divisible by attention heads."
            )

        if mlp_ratio <= 0:

            raise ValueError(
                "MLP ratio must be positive."
            )

        if not 0 <= dropout < 1:

            raise ValueError(
                "Dropout must be in [0, 1)."
            )

        if not 0 <= attention_dropout < 1:

            raise ValueError(
                "Attention dropout must be in [0, 1)."
            )

        # ----------------------------------------------------
        # Pretrained protection
        # ----------------------------------------------------

        if pretrained:

            raise ValueError(
                "This custom SmallViT does not have a compatible "
                "pretrained checkpoint. Use pretrained=False."
            )

        # ----------------------------------------------------
        # Save architecture configuration
        # ----------------------------------------------------

        self.num_classes = num_classes

        self.image_size = image_size

        self.patch_size = patch_size

        self.in_channels = in_channels

        self.embed_dim = embed_dim

        self.depth = depth

        self.num_heads = num_heads

        self.mlp_ratio = mlp_ratio

        self.dropout_rate = dropout

        self.attention_dropout = attention_dropout

        self.freeze_backbone = freeze_backbone

        self.pretrained = False

        # ----------------------------------------------------
        # Patch configuration
        # ----------------------------------------------------

        self.num_patches_per_side = (
            image_size // patch_size
        )

        self.num_patches = (
            self.num_patches_per_side ** 2
        )

        self.sequence_length = (
            self.num_patches + 1
        )

        # ----------------------------------------------------
        # Patch embedding
        # ----------------------------------------------------

        self.patch_embed = nn.Conv2d(

            in_channels=in_channels,

            out_channels=embed_dim,

            kernel_size=patch_size,

            stride=patch_size,

        )

        # ----------------------------------------------------
        # Class token
        # ----------------------------------------------------

        self.cls_token = nn.Parameter(

            torch.zeros(
                1,
                1,
                embed_dim
            )

        )

        # ----------------------------------------------------
        # Positional embedding
        # ----------------------------------------------------

        self.pos_embed = nn.Parameter(

            torch.zeros(

                1,

                self.sequence_length,

                embed_dim

            )

        )

        self.pos_dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Transformer encoder
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(

            [

                TransformerBlock(

                    embed_dim=embed_dim,

                    num_heads=num_heads,

                    mlp_ratio=mlp_ratio,

                    dropout=dropout,

                    attention_dropout=attention_dropout,

                )

                for _ in range(depth)

            ]

        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(
            embed_dim,
            eps=1e-6
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.head_dropout = nn.Dropout(
            dropout
        )

        self.head = nn.Linear(

            embed_dim,

            num_classes

        )

        # ----------------------------------------------------
        # Initialize weights
        # ----------------------------------------------------

        self._initialize_weights()

        # ----------------------------------------------------
        # Freeze backbone if requested
        # ----------------------------------------------------

        if freeze_backbone:

            self._freeze_backbone()

    # ========================================================
    # WEIGHT INITIALIZATION
    # ========================================================

    def _initialize_weights(self):

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02
        )

        nn.init.kaiming_normal_(
            self.patch_embed.weight,
            mode="fan_out",
            nonlinearity="linear"
        )

        if self.patch_embed.bias is not None:

            nn.init.zeros_(
                self.patch_embed.bias
            )

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(module, nn.LayerNorm):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    # ========================================================
    # FREEZE BACKBONE
    # ========================================================

    def _freeze_backbone(self):

        backbone_modules = [

            self.patch_embed,

            self.blocks,

            self.norm,

        ]

        for module in backbone_modules:

            for parameter in module.parameters():

                parameter.requires_grad = False

        self.cls_token.requires_grad = False

        self.pos_embed.requires_grad = False

        # Classification head remains trainable.

        for parameter in self.head.parameters():

            parameter.requires_grad = True

    # ========================================================
    # PATCH EMBEDDING
    # ========================================================

    def forward_patch_embedding(self, x):

        x = self.patch_embed(x)

        # [B, D, H/P, W/P]

        x = x.flatten(2)

        # [B, D, N] -> [B, N, D]

        x = x.transpose(1, 2)

        return x

    # ========================================================
    # TRANSFORMER TOKEN EXTRACTION
    # ========================================================

    def forward_tokens(self, x):

        batch_size = x.shape[0]

        # ----------------------------------------------------
        # Patch embedding
        # ----------------------------------------------------

        x = self.forward_patch_embedding(x)

        # ----------------------------------------------------
        # Add class token
        # ----------------------------------------------------

        cls_tokens = self.cls_token.expand(

            batch_size,

            -1,

            -1

        )

        x = torch.cat(

            (cls_tokens, x),

            dim=1

        )

        # ----------------------------------------------------
        # Add positional embeddings
        # ----------------------------------------------------

        x = x + self.pos_embed

        x = self.pos_dropout(x)

        # ----------------------------------------------------
        # Transformer encoder
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.norm(x)

        return x

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def forward_features(self, x):

        tokens = self.forward_tokens(x)

        # Extract class token representation.

        cls_features = tokens[:, 0]

        return cls_features

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
                f"got {x.shape[1]}."
            )

        if (
            x.shape[2] != self.image_size
            or x.shape[3] != self.image_size
        ):

            raise ValueError(
                f"Expected image size {self.image_size}x"
                f"{self.image_size}, received "
                f"{x.shape[2]}x{x.shape[3]}."
            )

        features = self.forward_features(x)

        logits = self.head(

            self.head_dropout(features)

        )

        return logits

    # ========================================================
    # GRAD-CAM TARGET LAYER
    # ========================================================

    def get_target_layer(self):

        return self.blocks[-1].norm2

    # ========================================================
    # XAI TOKEN RESHAPE
    # ========================================================

    def reshape_transform(self, tensor):

        """
        Convert transformer token output into spatial feature maps.

        Input:
            [B, 197, 192]

        Output:
            [B, 192, 14, 14]

        The first token is the classification token and is excluded.
        """

        if tensor.ndim != 3:

            raise ValueError(
                "Expected transformer tokens [B, N, D]."
            )

        if tensor.shape[1] != self.sequence_length:

            raise ValueError(
                "Unexpected transformer sequence length."
            )

        batch_size = tensor.shape[0]

        patch_tokens = tensor[:, 1:, :]

        spatial = patch_tokens.reshape(

            batch_size,

            self.num_patches_per_side,

            self.num_patches_per_side,

            self.embed_dim

        )

        spatial = spatial.permute(
            0, 3, 1, 2
        )

        return spatial

    # ========================================================
    # ARCHITECTURE CONFIGURATION
    # ========================================================

    def get_architecture_config(self):

        return {

            "model_name": "SmallViT",

            "architecture_type": "Vision Transformer",

            "pretrained": False,

            "pretrained_source": "None - trained from scratch",

            "input_channels": self.in_channels,

            "image_size": self.image_size,

            "patch_size": self.patch_size,

            "patches_per_side": self.num_patches_per_side,

            "total_patches": self.num_patches,

            "sequence_length": self.sequence_length,

            "class_token": True,

            "positional_embedding": "Learnable",

            "embedding_dimension": self.embed_dim,

            "transformer_depth": self.depth,

            "attention_heads": self.num_heads,

            "mlp_ratio": self.mlp_ratio,

            "normalization": "LayerNorm",

            "activation": "GELU",

            "num_classes": self.num_classes,

            "classifier": str(self.head),

            "dropout": self.dropout_rate,

            "attention_dropout": self.attention_dropout,

            "freeze_backbone": self.freeze_backbone,

            "gradcam_target_layer": (
                "blocks[-1].norm2"
            ),

            "gradcam_reshape": (
                "reshape_transform"
            ),

            "output_activation": "None (raw logits)",

            "torchvision_version": torchvision.__version__,

        }


# ============================================================
# MODEL BUILDER
# ============================================================

def build_small_vit(
    num_classes=NUM_CLASSES,
    pretrained=False,
    freeze_backbone=False,
    **kwargs
):

    return SmallViT(

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

    print("MAIZE LEAF - SMALL VISION TRANSFORMER TEST")

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
    # Build model
    # --------------------------------------------------------

    print("\nBuilding SmallViT from scratch...")

    model = build_small_vit(

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

    assert architecture["image_size"] == 224

    assert architecture["patch_size"] == 16

    assert architecture["total_patches"] == 196

    assert architecture["sequence_length"] == 197

    assert architecture["embedding_dimension"] == 192

    assert architecture["transformer_depth"] == 6

    assert architecture["attention_heads"] == 3

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
    # Input
    # --------------------------------------------------------

    inputs = torch.randn(

        2, 3, 224, 224,

        device=device

    )

    targets = torch.randint(

        0, NUM_CLASSES, (2,),

        device=device

    )

    # --------------------------------------------------------
    # Patch embedding
    # --------------------------------------------------------

    patches = model.forward_patch_embedding(inputs)

    assert patches.shape == (2, 196, 192)

    print(
        f"\n[PASS] Patch embedding: {tuple(patches.shape)}"
    )

    # --------------------------------------------------------
    # Transformer token output
    # --------------------------------------------------------

    model.train()

    tokens = model.forward_tokens(inputs)

    assert tokens.shape == (2, 197, 192)

    assert torch.isfinite(tokens).all()

    print(
        f"[PASS] Transformer tokens: {tuple(tokens.shape)}"
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    features = model.forward_features(inputs)

    assert features.shape == (2, 192)

    print(
        f"[PASS] Feature extraction: {tuple(features.shape)}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    outputs = model(inputs)

    assert outputs.shape == (2, NUM_CLASSES)

    assert torch.isfinite(outputs).all()

    print("\n[PASS] Forward propagation")

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

    assert isinstance(target_layer, nn.LayerNorm)

    print(
        f"[PASS] Grad-CAM target layer: "
        f"{target_layer.__class__.__name__}"
    )

    # --------------------------------------------------------
    # Grad-CAM reshape test
    # --------------------------------------------------------

    spatial_features = model.reshape_transform(tokens)

    assert spatial_features.shape == (
        2, 192, 14, 14
    )

    print(
        f"[PASS] Grad-CAM reshape: "
        f"{tuple(spatial_features.shape)}"
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

    frozen_model = build_small_vit(

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

        for parameter in frozen_model.head.parameters()

    )

    assert (
        frozen_info["trainable_parameters"]
        == classifier_parameters
    )

    assert all(

        not parameter.requires_grad

        for parameter in frozen_model.blocks.parameters()

    )

    assert not frozen_model.cls_token.requires_grad

    assert not frozen_model.pos_embed.requires_grad

    print("[PASS] Backbone parameter freezing")

    print("[PASS] Class token and positional embedding freezing")

    print("[PASS] Classifier remains trainable")

    # --------------------------------------------------------
    # Pretrained protection test
    # --------------------------------------------------------

    print("\nTesting pretrained protection...")

    try:

        build_small_vit(
            pretrained=True
        )

        raise AssertionError(
            "Unsupported pretrained weights were accepted."
        )

    except ValueError:

        print("[PASS] Unsupported pretrained protection")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    del frozen_model

    if device.type == "cuda":

        torch.cuda.empty_cache()

    print("\nSMALL VISION TRANSFORMER TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
