
"""
Maize Leaf Classification
Centralized Image Preprocessing Pipeline

Responsibilities:
1. Standardize image dimensions.
2. RGB image handling.
3. ImageNet normalization.
4. Configurable training augmentation.
5. Consistent validation/test preprocessing.
6. Reusable transformations for all seven models.

No random augmentation is applied to validation or test data.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.class_names import (
    PROJECT_ROOT,
    DATASET_DIR,
    CLASS_NAMES,
    SUPPORTED_EXTENSIONS,
)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# AUGMENTATION CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class AugmentationConfig:

    horizontal_flip: float = 0.5

    vertical_flip: float = 0.0

    rotation_degrees: float = 10.0

    brightness: float = 0.15

    contrast: float = 0.15

    saturation: float = 0.10

    hue: float = 0.02


# ============================================================
# PREPROCESSING CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PreprocessingConfig:

    image_size: int = DEFAULT_IMAGE_SIZE

    normalization: str = "imagenet"

    interpolation: str = "bilinear"

    augmentation: AugmentationConfig = AugmentationConfig()


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_config(config):

    if config.image_size <= 0:

        raise ValueError(
            "Image size must be greater than zero."
        )

    if config.normalization not in ("imagenet", "none"):

        raise ValueError(
            "Normalization must be 'imagenet' or 'none'."
        )

    if config.interpolation not in (
        "bilinear",
        "bicubic",
        "nearest"
    ):

        raise ValueError(
            "Unsupported interpolation method."
        )

    aug = config.augmentation

    for name, value in [
        ("horizontal_flip", aug.horizontal_flip),
        ("vertical_flip", aug.vertical_flip),
    ]:

        if not 0.0 <= value <= 1.0:

            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    if aug.rotation_degrees < 0:

        raise ValueError(
            "Rotation degrees cannot be negative."
        )

    for name, value in [
        ("brightness", aug.brightness),
        ("contrast", aug.contrast),
        ("saturation", aug.saturation),
    ]:

        if not 0.0 <= value <= 1.0:

            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    if not 0.0 <= aug.hue <= 0.5:

        raise ValueError(
            "Hue must be between 0 and 0.5."
        )

    return True


# ============================================================
# INTERPOLATION
# ============================================================

def get_interpolation(method):

    interpolation_map = {

        "bilinear": InterpolationMode.BILINEAR,

        "bicubic": InterpolationMode.BICUBIC,

        "nearest": InterpolationMode.NEAREST,

    }

    if method not in interpolation_map:

        raise ValueError(
            f"Unknown interpolation: {method}"
        )

    return interpolation_map[method]


# ============================================================
# NORMALIZATION
# ============================================================

def get_normalization(normalization):

    if normalization == "imagenet":

        return transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

    if normalization == "none":

        return torch.nn.Identity()

    raise ValueError(
        f"Unsupported normalization: {normalization}"
    )


# ============================================================
# TRAINING TRANSFORM
# ============================================================

def build_train_transform(
    config: Optional[PreprocessingConfig] = None,
    augmentation: bool = True
):

    if config is None:

        config = PreprocessingConfig()

    validate_config(config)

    aug = config.augmentation

    interpolation = get_interpolation(
        config.interpolation
    )

    pipeline = [

        transforms.Resize(
            (config.image_size, config.image_size),
            interpolation=interpolation,
            antialias=True
        ),

    ]

    if augmentation:

        if aug.horizontal_flip > 0:

            pipeline.append(
                transforms.RandomHorizontalFlip(
                    p=aug.horizontal_flip
                )
            )

        if aug.vertical_flip > 0:

            pipeline.append(
                transforms.RandomVerticalFlip(
                    p=aug.vertical_flip
                )
            )

        if aug.rotation_degrees > 0:

            pipeline.append(
                transforms.RandomRotation(
                    degrees=aug.rotation_degrees,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(0, 0, 0)
                )
            )

        if any([
            aug.brightness,
            aug.contrast,
            aug.saturation,
            aug.hue
        ]):

            pipeline.append(
                transforms.ColorJitter(
                    brightness=aug.brightness,
                    contrast=aug.contrast,
                    saturation=aug.saturation,
                    hue=aug.hue
                )
            )

    pipeline.extend([

        transforms.ToTensor(),

        get_normalization(config.normalization),

    ])

    return transforms.Compose(pipeline)


# ============================================================
# VALIDATION / TEST TRANSFORM
# ============================================================

def build_eval_transform(
    config: Optional[PreprocessingConfig] = None
):

    if config is None:

        config = PreprocessingConfig()

    validate_config(config)

    interpolation = get_interpolation(
        config.interpolation
    )

    return transforms.Compose([

        transforms.Resize(
            (config.image_size, config.image_size),
            interpolation=interpolation,
            antialias=True
        ),

        transforms.ToTensor(),

        get_normalization(config.normalization),

    ])


# ============================================================
# COMPLETE TRANSFORM FACTORY
# ============================================================

def build_transforms(
    image_size=DEFAULT_IMAGE_SIZE,
    augmentation=True,
    normalization="imagenet",
    augmentation_config=None,
    interpolation="bilinear"
):

    if augmentation_config is None:

        augmentation_config = AugmentationConfig()

    config = PreprocessingConfig(

        image_size=image_size,

        normalization=normalization,

        interpolation=interpolation,

        augmentation=augmentation_config,

    )

    validate_config(config)

    train_transform = build_train_transform(
        config=config,
        augmentation=augmentation
    )

    eval_transform = build_eval_transform(
        config=config
    )

    return train_transform, eval_transform


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def get_transforms(
    image_size=DEFAULT_IMAGE_SIZE,
    augmentation=True
):

    """
    Compatible with the previous dataset_loader interface.

    Returns:
        train_transform, evaluation_transform
    """

    return build_transforms(
        image_size=image_size,
        augmentation=augmentation,
        normalization="imagenet"
    )


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

def get_preprocessing_summary(
    image_size=DEFAULT_IMAGE_SIZE,
    augmentation=True,
    normalization="imagenet",
    augmentation_config=None
):

    if augmentation_config is None:

        augmentation_config = AugmentationConfig()

    config = PreprocessingConfig(

        image_size=image_size,

        normalization=normalization,

        augmentation=augmentation_config

    )

    validate_config(config)

    return {

        "image_size": image_size,

        "channels": 3,

        "color_space": "RGB",

        "normalization": normalization,

        "imagenet_mean": list(IMAGENET_MEAN)
        if normalization == "imagenet" else None,

        "imagenet_std": list(IMAGENET_STD)
        if normalization == "imagenet" else None,

        "interpolation": config.interpolation,

        "augmentation_enabled": augmentation,

        "augmentation": {

            "horizontal_flip": augmentation_config.horizontal_flip,

            "vertical_flip": augmentation_config.vertical_flip,

            "rotation_degrees": augmentation_config.rotation_degrees,

            "brightness": augmentation_config.brightness,

            "contrast": augmentation_config.contrast,

            "saturation": augmentation_config.saturation,

            "hue": augmentation_config.hue,

        } if augmentation else None,

        "random_crop": False,

        "validation_augmentation": False,

        "test_augmentation": False,

    }


# ============================================================
# EXECUTION TEST
# ============================================================

def main():

    print("=" * 70)

    print("MAIZE LEAF PREPROCESSING TEST")

    print("=" * 70)

    print(f"\nProject: {PROJECT_ROOT}")

    print(f"Dataset: {DATASET_DIR}")

    print(f"Classes: {len(CLASS_NAMES)}")

    print(f"Image Size: {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}")

    train_transform, eval_transform = build_transforms()

    print("\nTRAIN TRANSFORM")

    print(train_transform)

    print("\nEVALUATION TRANSFORM")

    print(eval_transform)

    print("\nTesting images from all seven classes...")

    for class_name in CLASS_NAMES:

        class_dir = DATASET_DIR / class_name

        images = sorted([

            path

            for path in class_dir.rglob("*")

            if path.is_file()

            and path.suffix.lower() in SUPPORTED_EXTENSIONS

        ])

        if not images:

            raise RuntimeError(
                f"No images found: {class_name}"
            )

        image_path = images[0]

        with Image.open(image_path) as image:

            rgb_image = image.convert("RGB")

            train_tensor = train_transform(rgb_image)

            eval_tensor = eval_transform(rgb_image)

            eval_tensor_repeat = eval_transform(rgb_image)

        expected_shape = (
            3,
            DEFAULT_IMAGE_SIZE,
            DEFAULT_IMAGE_SIZE
        )

        assert train_tensor.shape == expected_shape

        assert eval_tensor.shape == expected_shape

        assert torch.isfinite(train_tensor).all()

        assert torch.isfinite(eval_tensor).all()

        assert torch.equal(
            eval_tensor,
            eval_tensor_repeat
        ), "Evaluation transform is not deterministic."

        print(
            f"[PASS] {class_name:<40} "
            f"Shape: {tuple(eval_tensor.shape)}"
        )

    print("\nChecking normalization...")

    print(f"Mean: {IMAGENET_MEAN}")

    print(f"Std: {IMAGENET_STD}")

    print("\nPREPROCESSING TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
