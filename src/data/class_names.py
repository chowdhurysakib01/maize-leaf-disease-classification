
"""
Central class configuration for Maize Leaf Classification.

Project: D:/MaizeLeaf
Number of classes: 7
Potassium Deficiency: Excluded
"""

from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset"

# --------------------------------------------------
# OFFICIAL CLASS MAPPING
# --------------------------------------------------

CLASS_NAMES = (
    "Fall Armyworm (Insect) Damage",
    "Healthy Leaf",
    "Leaf Spot",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Northern Leaf Blight",
    "Phosphorus Deficiency",
)

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_IDX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

IDX_TO_CLASS = {
    index: class_name
    for index, class_name in enumerate(CLASS_NAMES)
}

# --------------------------------------------------
# IMAGE CONFIGURATION
# --------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# --------------------------------------------------
# DATASET VALIDATION
# --------------------------------------------------

def validate_dataset_folders():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    actual_classes = {
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    }

    expected_classes = set(CLASS_NAMES)

    missing_classes = expected_classes - actual_classes
    unexpected_classes = actual_classes - expected_classes

    if missing_classes or unexpected_classes:

        raise ValueError(
            "\nDataset class mismatch detected!\n"
            f"Missing classes: {sorted(missing_classes)}\n"
            f"Unexpected classes: {sorted(unexpected_classes)}"
        )

    return True


def get_dataset_class_counts():

    validate_dataset_folders()

    class_counts = {}

    for class_name in CLASS_NAMES:

        class_dir = DATASET_DIR / class_name

        image_files = [
            file
            for file in class_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        class_counts[class_name] = len(image_files)

    return class_counts


# --------------------------------------------------
# EXECUTION TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("MAIZE LEAF CLASSIFICATION - CLASS CONFIGURATION")
    print("=" * 60)

    print(f"\nProject Root : {PROJECT_ROOT}")
    print(f"Dataset Path: {DATASET_DIR}")
    print(f"Total Classes: {NUM_CLASSES}\n")

    counts = get_dataset_class_counts()

    total_images = 0

    for index, (class_name, count) in enumerate(counts.items()):

        print(f"{index}: {class_name:<40} {count:>6}")

        total_images += count

    print("-" * 60)
    print(f"TOTAL IMAGES: {total_images}")
    print("=" * 60)

    print("\nCLASS CONFIGURATION: PASSED")
