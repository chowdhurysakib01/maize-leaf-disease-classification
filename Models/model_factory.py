
"""
Maize Leaf Classification
Centralized Model Factory

Supported architectures:
    1. Custom CNN
    2. ResNet50
    3. EfficientNet-B0
    4. MobileNetV3-Large
    5. DenseNet121
    6. ConvNeXt-Tiny
    7. Small ViT
    8. RiceLeaf-DMNet (MobileNetV3-Large + DenseNet121)
    9. MobNetFine (MobileNetV3-Large + MSAF)

Responsibilities:
    - Centralized model selection.
    - Model initialization.
    - Pretrained weight configuration.
    - Backbone freezing.
    - Parameter information.
    - Architecture metadata.
    - Grad-CAM target layer access.
    - ViT reshape transform access.
    - Training mode management.

All models output raw logits.
"""

import warnings
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.data.class_names import NUM_CLASSES

from Models.custom_cnn import (
    build_custom_cnn,
)

from Models.resnet50 import (
    build_resnet50,
)

from Models.efficientnet_b0 import (
    build_efficientnet_b0,
)

from Models.mobilenet_v3 import (
    build_mobilenet_v3,
)

from Models.MobNetFine import (
    build_mobnetfine,
)

from Models.densenet121 import (
    build_densenet121,
)

from Models.convnext_tiny import (
    build_convnext_tiny,
)

from Models.small_vit import (
    build_small_vit,
)


# The requested filename contains a hyphen, so load it through importlib
# instead of a normal Python import statement.
def _load_riceleaf_dmnet_builder():
    model_path = Path(__file__).resolve().with_name("RiceLeaf-DMNet.py")
    module_name = "Models._riceleaf_dmnet_dynamic"

    spec = importlib.util.spec_from_file_location(
        module_name,
        model_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load RiceLeaf-DMNet model file: {model_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    builder = getattr(module, "build_riceleaf_dmnet", None)
    if not callable(builder):
        raise ImportError(
            "RiceLeaf-DMNet.py must define build_riceleaf_dmnet()."
        )
    return builder


build_riceleaf_dmnet = _load_riceleaf_dmnet_builder()


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {

    "custom_cnn": {

        "builder": build_custom_cnn,

        "display_name": "Custom CNN",

        "pretrained_supported": False,

        "freeze_supported": True,

    },

    "resnet50": {

        "builder": build_resnet50,

        "display_name": "ResNet50",

        "pretrained_supported": True,

        "freeze_supported": True,

    },

    "efficientnet_b0": {

        "builder": build_efficientnet_b0,

        "display_name": "EfficientNet-B0",

        "pretrained_supported": True,

        "freeze_supported": True,

    },

    "mobilenet_v3": {

        "builder": build_mobilenet_v3,

        "display_name": "MobileNetV3-Large",

        "pretrained_supported": True,

        "freeze_supported": True,

    },

    "mobnetfine": {

        "builder": build_mobnetfine,

        "display_name": "MobNetFine (MobileNetV3-Large + MSAF)",

        "pretrained_supported": True,

        "freeze_supported": True,

    },

    "densenet121": {

        "builder": build_densenet121,

        "display_name": "DenseNet121",

        "pretrained_supported": True,

        "freeze_supported": True,

    },

    "convnext_tiny": {

        "builder": build_convnext_tiny,

        "display_name": "ConvNeXt-Tiny",

        "pretrained_supported": True,

        "freeze_supported": True,

    },

    "small_vit": {

        "builder": build_small_vit,

        "display_name": "Small Vision Transformer",

        "pretrained_supported": False,

        "freeze_supported": True,

    },

    "riceleaf_dmnet": {

        "builder": build_riceleaf_dmnet,

        "display_name": "RiceLeaf-DMNet (MobileNetV3-Large + DenseNet121)",

        "pretrained_supported": True,

        "freeze_supported": True,

    },

}


# ============================================================
# MODEL NAME VALIDATION
# ============================================================

def normalize_model_name(model_name):

    if not isinstance(model_name, str):

        raise TypeError(
            "Model name must be a string."
        )

    model_name = model_name.strip().lower()

    if model_name not in MODEL_REGISTRY:

        available = ", ".join(
            MODEL_REGISTRY.keys()
        )

        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available}"
        )

    return model_name


# ============================================================
# AVAILABLE MODELS
# ============================================================

def get_available_models():

    return list(MODEL_REGISTRY.keys())


# ============================================================
# CLASSIFIER MODULE ACCESS
# ============================================================

def get_classifier_module(model_name, model):

    model_name = normalize_model_name(model_name)

    if model_name == "custom_cnn":

        return model.classifier

    if model_name == "resnet50":

        return model.backbone.fc

    if model_name in (
        "efficientnet_b0",
        "mobilenet_v3",
        "densenet121",
        "convnext_tiny",
    ):

        return model.backbone.classifier

    if model_name == "small_vit":

        return model.head

    if model_name == "mobnetfine":

        return model.classifier

    if model_name == "riceleaf_dmnet":

        return model.classifier

    raise ValueError(
        f"Classifier access unavailable for {model_name}."
    )


# ============================================================
# MODEL BUILDING
# ============================================================

def build_model(
    model_name,
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=False,
    **model_kwargs
):
    """
    Build one selected model.

    Args:
        model_name:
            One of the registered model names.

        num_classes:
            Number of output classes.

        pretrained:
            Use ImageNet weights where supported.

        freeze_backbone:
            Freeze feature extraction layers.

        model_kwargs:
            Additional architecture-specific arguments.

    Returns:
        Initialized PyTorch model.
    """

    model_name = normalize_model_name(model_name)

    if num_classes <= 1:

        raise ValueError(
            "Number of classes must be greater than 1."
        )

    if not isinstance(pretrained, bool):

        raise TypeError(
            "pretrained must be boolean."
        )

    if not isinstance(freeze_backbone, bool):

        raise TypeError(
            "freeze_backbone must be boolean."
        )

    model_info = MODEL_REGISTRY[model_name]

    builder = model_info["builder"]

    # --------------------------------------------------------
    # Pretrained compatibility
    # --------------------------------------------------------

    actual_pretrained = pretrained

    if (
        pretrained
        and not model_info["pretrained_supported"]
    ):

        warnings.warn(

            f"{model_info['display_name']} does not "
            "support pretrained weights in this pipeline. "
            "Initializing from scratch.",

            RuntimeWarning

        )

        actual_pretrained = False

    # --------------------------------------------------------
    # Initialize architecture
    # --------------------------------------------------------

    if model_name == "custom_cnn":

        model = builder(

            num_classes=num_classes,

            **model_kwargs

        )

    else:

        model = builder(

            num_classes=num_classes,

            pretrained=actual_pretrained,

            freeze_backbone=freeze_backbone,

            **model_kwargs

        )

    # --------------------------------------------------------
    # Custom CNN backbone freezing
    # --------------------------------------------------------

    if model_name == "custom_cnn" and freeze_backbone:

        for parameter in model.features.parameters():

            parameter.requires_grad = False

    # --------------------------------------------------------
    # Save factory metadata
    # --------------------------------------------------------

    model.factory_model_name = model_name

    model.factory_display_name = model_info["display_name"]

    model.factory_pretrained_requested = pretrained

    model.factory_pretrained_used = actual_pretrained

    model.factory_freeze_backbone = freeze_backbone

    model.factory_num_classes = num_classes

    # --------------------------------------------------------
    # Verify classifier output
    # --------------------------------------------------------

    classifier = get_classifier_module(
        model_name,
        model
    )

    linear_layers = [

        module

        for module in classifier.modules()

        if isinstance(module, nn.Linear)

    ]

    if not linear_layers:

        raise RuntimeError(
            "No Linear classification layer found."
        )

    if linear_layers[-1].out_features != num_classes:

        raise RuntimeError(
            "Classifier output does not match num_classes."
        )

    # --------------------------------------------------------
    # Verify trainable parameters
    # --------------------------------------------------------

    if not any(
        parameter.requires_grad
        for parameter in model.parameters()
    ):

        raise RuntimeError(
            "Model contains no trainable parameters."
        )

    if freeze_backbone:

        if not any(
            parameter.requires_grad
            for parameter in classifier.parameters()
        ):

            raise RuntimeError(
                "Classifier must remain trainable "
                "when the backbone is frozen."
            )

    return model


# ============================================================
# BUILD MODEL FROM TRAINING CONFIGURATION
# ============================================================

def build_model_from_config(
    config,
    **model_kwargs
):
    """
    Initialize a model directly from TrainingConfig.

    Compatible with:
        configs.training_config

    Uses:
        config.model
        config.pretrained
        config.freeze_backbone
    """

    required_attributes = [
        "model",
        "pretrained",
        "freeze_backbone",
    ]

    for attribute in required_attributes:

        if not hasattr(config, attribute):

            raise AttributeError(
                f"Training configuration is missing: "
                f"{attribute}"
            )

    return build_model(

        model_name=config.model,

        num_classes=getattr(
            config,
            "num_classes",
            NUM_CLASSES
        ),

        pretrained=config.pretrained,

        freeze_backbone=config.freeze_backbone,

        **model_kwargs

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

    total_buffers = sum(

        buffer.numel()

        for buffer in model.buffers()

    )

    return {

        "total_parameters": total_parameters,

        "trainable_parameters": trainable_parameters,

        "frozen_parameters": frozen_parameters,

        "non_trainable_percentage": (

            100.0 * frozen_parameters / total_parameters

            if total_parameters else 0.0

        ),

        "trainable_percentage": (

            100.0 * trainable_parameters / total_parameters

            if total_parameters else 0.0

        ),

        "buffer_elements": total_buffers,

    }


# ============================================================
# MODEL METADATA
# ============================================================

def get_model_metadata(model):

    model_name = getattr(
        model,
        "factory_model_name",
        None
    )

    if model_name is None:

        raise ValueError(
            "Model must be initialized using build_model()."
        )

    architecture = model.get_architecture_config()

    parameters = get_parameter_information(model)

    return {

        "model_name": model_name,

        "display_name": model.factory_display_name,

        "num_classes": model.factory_num_classes,

        "pretrained_requested": (
            model.factory_pretrained_requested
        ),

        "pretrained_used": (
            model.factory_pretrained_used
        ),

        "freeze_backbone": (
            model.factory_freeze_backbone
        ),

        "architecture": architecture,

        "parameters": parameters,

    }


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

def get_gradcam_target_layer(model):

    model_name = getattr(
        model,
        "factory_model_name",
        None
    )

    if model_name is None:

        raise ValueError(
            "Model was not initialized using build_model()."
        )

    # Custom CNN does not expose a dedicated method.

    if model_name == "custom_cnn":

        return model.features[-1].block[3]

    return model.get_target_layer()


# ============================================================
# GRAD-CAM RESHAPE FUNCTION
# ============================================================

def get_gradcam_reshape_transform(model):

    if hasattr(model, "reshape_transform"):

        return model.reshape_transform

    return None


# ============================================================
# TRAINING MODE MANAGEMENT
# ============================================================

def set_model_training_mode(model, training=True):
    """
    Standardized training/evaluation mode management.

    Important:
        Custom CNN frozen BatchNorm layers must remain
        in evaluation mode when backbone freezing is enabled.

    Other supported architectures already implement
    their own frozen BatchNorm handling where applicable.
    """

    model.train(training)

    if (
        training
        and getattr(
            model,
            "factory_model_name",
            None
        ) == "custom_cnn"
        and getattr(
            model,
            "factory_freeze_backbone",
            False
        )
    ):

        for module in model.features.modules():

            if isinstance(module, nn.BatchNorm2d):

                module.eval()

    return model


# ============================================================
# PRINT MODEL REPORT
# ============================================================

def print_model_report(model):

    metadata = get_model_metadata(model)

    print("\n" + "=" * 70)

    print("MAIZE LEAF - MODEL FACTORY REPORT")

    print("=" * 70)

    print(
        f"Model name                 : "
        f"{metadata['model_name']}"
    )

    print(
        f"Display name               : "
        f"{metadata['display_name']}"
    )

    print(
        f"Number of classes          : "
        f"{metadata['num_classes']}"
    )

    print(
        f"Pretrained requested       : "
        f"{metadata['pretrained_requested']}"
    )

    print(
        f"Pretrained actually used   : "
        f"{metadata['pretrained_used']}"
    )

    print(
        f"Backbone frozen            : "
        f"{metadata['freeze_backbone']}"
    )

    print("\nPARAMETER INFORMATION")

    for key, value in metadata["parameters"].items():

        if isinstance(value, float):

            print(f"{key:<30}: {value:.2f}%")

        else:

            print(f"{key:<30}: {value:,}")

    print("\nARCHITECTURE INFORMATION")

    for key, value in metadata["architecture"].items():

        print(f"{key:<30}: {value}")

    print("=" * 70)


# ============================================================
# FACTORY TEST
# ============================================================

def main():

    print("=" * 70)

    print("MAIZE LEAF - CENTRAL MODEL FACTORY TEST")

    print("=" * 70)

    print("\nAvailable models:")

    for index, name in enumerate(
        get_available_models(),
        start=1
    ):

        print(f"{index}. {name}")

    # --------------------------------------------------------
    # Test all seven architectures
    # --------------------------------------------------------

    print("\nTESTING ALL REGISTERED MODELS")

    for model_name in get_available_models():

        print("\n" + "-" * 70)

        print(f"Testing: {model_name}")

        # CPU is deliberately used for this registry test.
        # Full GPU forward/backward tests were already performed
        # individually for all seven architectures.

        model = build_model(

            model_name=model_name,

            num_classes=NUM_CLASSES,

            pretrained=False,

            freeze_backbone=False,

        )

        # Verify seven-class configuration.

        assert model.factory_num_classes == 7

        # Verify output classifier.

        classifier = get_classifier_module(
            model_name,
            model
        )

        linear_layers = [

            module

            for module in classifier.modules()

            if isinstance(module, nn.Linear)

        ]

        assert linear_layers[-1].out_features == 7

        # Verify parameter count.

        info = get_parameter_information(model)

        assert info["total_parameters"] > 0

        assert (
            info["total_parameters"]
            ==
            info["trainable_parameters"]
            +
            info["frozen_parameters"]
        )

        assert info["trainable_parameters"] > 0

        # Verify Grad-CAM layer.

        target_layer = get_gradcam_target_layer(
            model
        )

        assert isinstance(target_layer, nn.Module)

        # Verify metadata.

        metadata = get_model_metadata(model)

        assert metadata["num_classes"] == 7

        assert metadata["pretrained_used"] is False

        print(
            f"[PASS] Architecture initialization: "
            f"{metadata['display_name']}"
        )

        print(
            f"[PASS] Total parameters: "
            f"{info['total_parameters']:,}"
        )

        print(
            f"[PASS] Seven-class classifier"
        )

        print(
            f"[PASS] Grad-CAM target: "
            f"{target_layer.__class__.__name__}"
        )

        del model

    # --------------------------------------------------------
    # Test frozen backbone for all seven models
    # --------------------------------------------------------

    print("\nTESTING BACKBONE FREEZING FOR ALL REGISTERED MODELS")

    for model_name in get_available_models():

        model = build_model(

            model_name=model_name,

            num_classes=NUM_CLASSES,

            pretrained=False,

            freeze_backbone=True,

        )

        info = get_parameter_information(model)

        classifier = get_classifier_module(
            model_name,
            model
        )

        assert info["frozen_parameters"] > 0

        assert all(

            parameter.requires_grad

            for parameter in classifier.parameters()

        )

        # Verify factory training-mode management.

        set_model_training_mode(
            model,
            training=True
        )

        if model_name == "custom_cnn":

            for module in model.features.modules():

                if isinstance(module, nn.BatchNorm2d):

                    assert not module.training

        print(
            f"[PASS] Backbone freezing: {model_name}"
        )

        del model

    # --------------------------------------------------------
    # Test invalid model
    # --------------------------------------------------------

    print("\nTESTING INVALID MODEL PROTECTION")

    try:

        build_model(
            model_name="unknown_model",
            pretrained=False
        )

        raise AssertionError(
            "Invalid model name was accepted."
        )

    except ValueError:

        print("[PASS] Invalid model protection")

    # --------------------------------------------------------
    # Test pretrained compatibility
    # --------------------------------------------------------

    print("\nTESTING SCRATCH MODEL PROTECTION")

    for model_name in ("custom_cnn", "small_vit"):

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            model = build_model(

                model_name=model_name,

                pretrained=True

            )

        assert model.factory_pretrained_requested is True

        assert model.factory_pretrained_used is False

        print(
            f"[PASS] Scratch-only protection: {model_name}"
        )

        del model

    # --------------------------------------------------------
    # Test ViT reshape access
    # --------------------------------------------------------

    print("\nTESTING VISION TRANSFORMER XAI ACCESS")

    vit = build_model(
        "small_vit",
        pretrained=False
    )

    reshape = get_gradcam_reshape_transform(vit)

    assert callable(reshape)

    print("[PASS] ViT reshape transform access")

    del vit

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\nMODEL FACTORY TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
