
"""
Maize Leaf Classification - Dataset Loader
Project: D:/MaizeLeaf

Responsibilities:
1. Load fixed train/validation/test CSV manifests.
2. Validate class names and image paths.
3. Prevent image overlap across splits.
4. Check group-level leakage when group IDs exist.
5. Build PyTorch datasets and DataLoaders.
6. Apply consistent preprocessing and optional augmentation.

Expected CSV columns:
    relative_path, class_name, group_id, split

group_id may be empty if unavailable.
"""

from pathlib import Path
from collections import Counter
import csv
import random
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.class_names import (
    PROJECT_ROOT,
    DATASET_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    NUM_CLASSES,
    SUPPORTED_EXTENSIONS,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SPLIT_DIR = PROJECT_ROOT / "data_splits"

SPLIT_FILES = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}

DEFAULT_IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageFile.LOAD_TRUNCATED_IMAGES = False


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)


# --------------------------------------------------
# MANIFEST UTILITIES
# --------------------------------------------------

def get_column(row, possible_names, required=True):

    for name in possible_names:

        value = row.get(name)

        if value is not None and str(value).strip():

            return str(value).strip()

    if required:

        raise ValueError(
            f"Required CSV column missing. "
            f"Expected one of: {possible_names}"
        )

    return ""


def resolve_image_path(raw_path, class_name):

    path = Path(raw_path)

    candidates = []

    if path.is_absolute():

        candidates.append(path)

    else:

        candidates.extend([
            PROJECT_ROOT / path,
            DATASET_DIR / path,
            DATASET_DIR / class_name / path,
        ])

    for candidate in candidates:

        if candidate.is_file():

            return candidate.resolve()

    raise FileNotFoundError(
        f"Image not found: {raw_path}\n"
        f"Class: {class_name}"
    )


def load_split_csv(split_name):

    if split_name not in SPLIT_FILES:

        raise ValueError(
            f"Invalid split name: {split_name}"
        )

    csv_path = SPLIT_FILES[split_name]

    if not csv_path.exists():

        raise FileNotFoundError(
            f"\nSplit manifest not found: {csv_path}\n"
            "Create the fixed dataset splits before loading data."
        )

    records = []

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:

            raise ValueError(
                f"Empty CSV manifest: {csv_path}"
            )

        for line_number, row in enumerate(reader, start=2):

            raw_path = get_column(
                row,
                ["relative_path", "image_path", "filepath", "path"]
            )

            class_name = get_column(
                row,
                ["class_name", "class", "label"]
            )

            group_id = get_column(
                row,
                ["group_id", "plant_id", "session_id"],
                required=False
            )

            declared_split = get_column(
                row,
                ["split"],
                required=False
            )

            if declared_split and declared_split.lower() != split_name:
                raise ValueError(
                    f"Split mismatch at {csv_path}, "
                    f"line {line_number}: "
                    f"expected {split_name}, got {declared_split}"
                )

            if class_name not in CLASS_TO_IDX:

                raise ValueError(
                    f"Unknown class at line {line_number}: "
                    f"{class_name}"
                )

            image_path = resolve_image_path(
                raw_path,
                class_name
            )

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:

                raise ValueError(
                    f"Unsupported image extension: {image_path}"
                )

            records.append({
                "path": image_path,
                "relative_path": str(image_path.relative_to(PROJECT_ROOT)),
                "class_name": class_name,
                "label": CLASS_TO_IDX[class_name],
                "group_id": group_id,
                "split": split_name,
            })

    if not records:

        raise ValueError(
            f"No image records found in {csv_path}"
        )

    return records


# --------------------------------------------------
# SPLIT VALIDATION
# --------------------------------------------------

def validate_split_records(split_records):

    all_paths = {}

    all_groups = {}

    group_ids_available = False

    for split_name, records in split_records.items():

        for record in records:

            image_path = record["relative_path"]

            if image_path in all_paths:

                previous_split = all_paths[image_path]

                raise ValueError(
                    "\nDATA LEAKAGE DETECTED!\n"
                    f"Image: {image_path}\n"
                    f"First split: {previous_split}\n"
                    f"Second split: {split_name}"
                )

            all_paths[image_path] = split_name

            group_id = record["group_id"]

            if group_id:

                group_ids_available = True

                if group_id in all_groups:

                    previous_split = all_groups[group_id]

                    if previous_split != split_name:

                        raise ValueError(
                            "\nGROUP-LEVEL DATA LEAKAGE DETECTED!\n"
                            f"Group ID: {group_id}\n"
                            f"First split: {previous_split}\n"
                            f"Second split: {split_name}"
                        )

                else:

                    all_groups[group_id] = split_name

        counts = Counter(
            record["class_name"]
            for record in records
        )

        missing_classes = [
            name
            for name in CLASS_NAMES
            if counts.get(name, 0) == 0
        ]

        if missing_classes:

            raise ValueError(
                f"Missing classes in {split_name}: "
                f"{missing_classes}"
            )

    if not group_ids_available:

        warnings.warn(
            "No group IDs found in split manifests. "
            "Image-level overlap is checked, but group-level "
            "leakage cannot be independently verified.",
            RuntimeWarning
        )

    return True


def load_all_split_records():

    split_records = {}

    for split_name in ("train", "validation", "test"):

        split_records[split_name] = load_split_csv(split_name)

    validate_split_records(split_records)

    return split_records


# --------------------------------------------------
# IMAGE TRANSFORMATIONS
# --------------------------------------------------

def get_transforms(
    image_size=DEFAULT_IMAGE_SIZE,
    augmentation=True
):

    if image_size <= 0:

        raise ValueError("Image size must be positive.")

    train_transforms = []

    if augmentation:

        train_transforms.extend([

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(
                degrees=10,
                interpolation=InterpolationMode.BILINEAR
            ),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02
            ),

        ])

    train_transforms.extend([

        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

    ])

    evaluation_transforms = transforms.Compose([

        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),

    ])

    return (
        transforms.Compose(train_transforms),
        evaluation_transforms
    )


# --------------------------------------------------
# PYTORCH DATASET
# --------------------------------------------------

class MaizeLeafDataset(Dataset):

    def __init__(
        self,
        records,
        transform=None,
        return_path=False
    ):

        self.records = records

        self.transform = transform

        self.return_path = return_path

        if not self.records:

            raise ValueError("Dataset cannot be empty.")

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = record["path"]

        try:

            with Image.open(image_path) as image:

                image = image.convert("RGB")

                if self.transform is not None:

                    image = self.transform(image)

        except Exception as error:

            raise RuntimeError(
                f"Failed to load image: {image_path}\n"
                f"Original error: {error}"
            ) from error

        label = record["label"]

        if self.return_path:

            return image, label, str(image_path)

        return image, label


# --------------------------------------------------
# DATALOADER FACTORY
# --------------------------------------------------

def create_dataloaders(
    batch_size=16,
    image_size=224,
    num_workers=0,
    seed=42,
    augmentation=True,
    pin_memory=None,
    return_paths=False
):

    if batch_size <= 0:

        raise ValueError("Batch size must be positive.")

    if num_workers < 0:

        raise ValueError("num_workers cannot be negative.")

    split_records = load_all_split_records()

    train_transform, eval_transform = get_transforms(
        image_size=image_size,
        augmentation=augmentation
    )

    datasets = {

        "train": MaizeLeafDataset(
            split_records["train"],
            transform=train_transform,
            return_path=return_paths
        ),

        "validation": MaizeLeafDataset(
            split_records["validation"],
            transform=eval_transform,
            return_path=return_paths
        ),

        "test": MaizeLeafDataset(
            split_records["test"],
            transform=eval_transform,
            return_path=return_paths
        ),

    }

    if pin_memory is None:

        pin_memory = torch.cuda.is_available()

    generator = torch.Generator()

    generator.manual_seed(seed)

    dataloaders = {

        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=num_workers > 0,
        ),

        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
            persistent_workers=num_workers > 0,
        ),

        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
            persistent_workers=num_workers > 0,
        ),

    }

    metadata = {

        "num_classes": NUM_CLASSES,

        "class_names": list(CLASS_NAMES),

        "class_to_idx": CLASS_TO_IDX,

        "image_size": image_size,

        "batch_size": batch_size,

        "seed": seed,

        "augmentation": augmentation,

        "counts": {

            split_name: len(dataset)

            for split_name, dataset in datasets.items()

        },

    }

    return dataloaders, datasets, metadata


# --------------------------------------------------
# EXECUTION TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)

    print("MAIZE LEAF DATASET LOADER")

    print("=" * 65)

    print(f"Project: {PROJECT_ROOT}")

    print(f"Dataset: {DATASET_DIR}")

    print(f"Classes: {NUM_CLASSES}")

    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():

        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nChecking split manifests...")

    records = load_all_split_records()

    for split_name, split_data in records.items():

        print(f"\n{split_name.upper()}")

        print(f"Total images: {len(split_data)}")

        counts = Counter(
            item["class_name"]
            for item in split_data
        )

        for class_name in CLASS_NAMES:

            print(
                f"  {class_name:<40} "
                f"{counts[class_name]:>5}"
            )

    print("\nSPLIT VALIDATION: PASSED")

    print("=" * 65)

    print(
        "\nNext: Create the centralized split manifests "
        "before initializing DataLoaders."
    )
