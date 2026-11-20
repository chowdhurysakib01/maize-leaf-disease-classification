
"""
Maize Leaf Classification
Centralized Stratified Dataset Split Generator

Dataset:
    7 classes

Default split:
    Train      70%
    Validation 15%
    Test       15%

Random seed:
    42

Duplicate policy:
    Exact duplicates are detected using SHA-256.
    Same-class duplicates: keep one, exclude extra copies.
    Cross-class duplicates: stop split generation.

Original Dataset files are NEVER deleted.
"""

import argparse
import csv
import hashlib
import json
import math
import random

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.data.class_names import (
    PROJECT_ROOT,
    DATASET_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    NUM_CLASSES,
    SUPPORTED_EXTENSIONS,
    validate_dataset_folders,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "data_splits"

CSV_COLUMNS = [
    "relative_path",
    "class_name",
    "class_index",
    "group_id",
    "split",
    "width",
    "height",
    "sha256",
]

DUPLICATE_COLUMNS = [
    "sha256",
    "class_name",
    "kept_path",
    "excluded_path",
    "duplicate_group_size",
    "action",
]

DEFAULT_SEED = 42

DEFAULT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

CORE_SPLIT_FILES = [
    "split_manifest.csv",
    "train.csv",
    "validation.csv",
    "test.csv",
    "split_metadata.json",
]


# ============================================================
# HASHING
# ============================================================

def calculate_sha256(image_path):

    digest = hashlib.sha256()

    with open(image_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# IMAGE INTEGRITY
# ============================================================

def inspect_image(image_path):

    try:

        with Image.open(image_path) as image:

            width, height = image.size

            image.verify()

        if width <= 0 or height <= 0:

            raise ValueError("Invalid image dimensions.")

        return {
            "width": width,
            "height": height,
            "sha256": calculate_sha256(image_path),
        }

    except Exception as error:

        raise RuntimeError(
            f"Image integrity check failed:\n"
            f"{image_path}\n"
            f"Reason: {error}"
        ) from error


# ============================================================
# DATASET SCANNING
# ============================================================

def scan_dataset():

    print("\nScanning dataset...")

    validate_dataset_folders()

    records = []

    for class_name in CLASS_NAMES:

        class_dir = DATASET_DIR / class_name

        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not image_paths:

            raise RuntimeError(
                f"No supported images found: {class_name}"
            )

        print(
            f"\n{class_name}: {len(image_paths)} images"
        )

        for image_path in image_paths:

            info = inspect_image(image_path)

            records.append({

                "relative_path": image_path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),

                "class_name": class_name,

                "class_index": CLASS_TO_IDX[class_name],

                "group_id": "",

                "split": "",

                "width": info["width"],

                "height": info["height"],

                "sha256": info["sha256"],

            })

    print("\nImage integrity check: PASSED")

    print(f"Total scanned images: {len(records)}")

    return records


# ============================================================
# CSV WRITER
# ============================================================

def save_csv(records, output_path, columns):

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        writer.writerows(records)


# ============================================================
# EXACT DUPLICATE DETECTION
# ============================================================

def deduplicate_exact_images(records):

    print("\nChecking exact duplicates using SHA-256...")

    hash_groups = defaultdict(list)

    for record in records:

        hash_groups[record["sha256"]].append(record)

    retained = []

    duplicate_rows = []

    conflicts = []

    duplicate_group_count = 0

    for image_hash, group in hash_groups.items():

        if len(group) == 1:

            retained.append(group[0])

            continue

        duplicate_group_count += 1

        # Deterministic keeper selection.
        group = sorted(
            group,
            key=lambda item: item["relative_path"]
        )

        class_names = {
            item["class_name"]
            for item in group
        }

        # Same image assigned to different classes.
        if len(class_names) > 1:

            conflicts.append({
                "sha256": image_hash,
                "records": group,
            })

            for item in group:

                duplicate_rows.append({

                    "sha256": image_hash,

                    "class_name": item["class_name"],

                    "kept_path": "",

                    "excluded_path": item["relative_path"],

                    "duplicate_group_size": len(group),

                    "action": "cross_class_conflict",

                })

            continue

        # Same-class duplicates:
        # Keep first path, exclude all remaining copies.
        keeper = group[0]

        retained.append(keeper)

        for duplicate in group[1:]:

            duplicate_rows.append({

                "sha256": image_hash,

                "class_name": keeper["class_name"],

                "kept_path": keeper["relative_path"],

                "excluded_path": duplicate["relative_path"],

                "duplicate_group_size": len(group),

                "action": "excluded_from_split",

            })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Always preserve a duplicate audit report.
    save_csv(
        duplicate_rows,
        OUTPUT_DIR / "exact_duplicates.csv",
        DUPLICATE_COLUMNS
    )

    print(f"\nExact duplicate groups: {duplicate_group_count}")

    print(
        f"Extra copies excluded: {len(duplicate_rows) - sum(len(item['records']) for item in conflicts)}"
    )

    if conflicts:

        print("\nCROSS-CLASS DUPLICATE CONFLICTS DETECTED!")

        for conflict in conflicts:

            print(f"\nSHA256: {conflict['sha256']}")

            for item in conflict["records"]:

                print(
                    f"  {item['class_name']}: "
                    f"{item['relative_path']}"
                )

        raise RuntimeError(
            "The same exact image exists in different classes. "
            "Resolve the labeling conflict before splitting."
        )

    print("Same-class duplicate handling: PASSED")

    print("Original Dataset files: UNCHANGED")

    print(f"Images retained: {len(retained)}")

    return retained, duplicate_rows


# ============================================================
# SPLIT ALLOCATION
# ============================================================

def calculate_split_counts(number_of_images, ratios):

    split_names = [
        "train",
        "validation",
        "test"
    ]

    exact_counts = {
        name: number_of_images * ratios[name]
        for name in split_names
    }

    allocated = {
        name: math.floor(exact_counts[name])
        for name in split_names
    }

    remaining = number_of_images - sum(
        allocated.values()
    )

    priorities = sorted(
        split_names,
        key=lambda name: (
            -(exact_counts[name] - allocated[name]),
            split_names.index(name)
        )
    )

    for name in priorities[:remaining]:

        allocated[name] += 1

    return allocated


# ============================================================
# STRATIFIED RANDOM SPLIT
# ============================================================

def create_stratified_split(records, ratios, seed):

    print("\nCreating stratified random split...")

    rng = random.Random(seed)

    class_records = defaultdict(list)

    for record in records:

        class_records[record["class_name"]].append(
            record
        )

    split_records = {
        "train": [],
        "validation": [],
        "test": [],
    }

    class_split_counts = {}

    for class_name in CLASS_NAMES:

        images = class_records[class_name].copy()

        rng.shuffle(images)

        allocation = calculate_split_counts(
            len(images),
            ratios
        )

        train_end = allocation["train"]

        validation_end = (
            train_end + allocation["validation"]
        )

        assignments = {

            "train": images[:train_end],

            "validation": images[
                train_end:validation_end
            ],

            "test": images[validation_end:],

        }

        for split_name, assigned in assignments.items():

            for record in assigned:

                record["split"] = split_name

                split_records[split_name].append(
                    record
                )

        class_split_counts[class_name] = allocation

    for split_name in split_records:

        rng.shuffle(split_records[split_name])

    return split_records, class_split_counts


# ============================================================
# SPLIT VALIDATION
# ============================================================

def validate_generated_splits(
    split_records,
    original_records
):

    print("\nValidating final split assignments...")

    original_paths = {
        item["relative_path"]
        for item in original_records
    }

    assigned_paths = []

    assigned_hashes = set()

    for split_name, records in split_records.items():

        for record in records:

            if record["split"] != split_name:

                raise RuntimeError(
                    f"Incorrect assignment: "
                    f"{record['relative_path']}"
                )

            path = record["relative_path"]

            if path in assigned_paths:

                raise RuntimeError(
                    f"Image overlap detected: {path}"
                )

            assigned_paths.append(path)

            if record["sha256"] in assigned_hashes:

                raise RuntimeError(
                    "Exact duplicate image detected "
                    "after deduplication."
                )

            assigned_hashes.add(record["sha256"])

    if set(assigned_paths) != original_paths:

        raise RuntimeError(
            "Some retained images are missing or extra."
        )

    for split_name, records in split_records.items():

        present_classes = {
            record["class_name"]
            for record in records
        }

        if present_classes != set(CLASS_NAMES):

            raise RuntimeError(
                f"Missing class in {split_name}."
            )

    print("Image overlap check: PASSED")

    print("Duplicate exclusion check: PASSED")

    print("Seven-class consistency: PASSED")

    print("All retained images assigned: PASSED")


# ============================================================
# FINGERPRINT
# ============================================================

def create_fingerprint(records, include_split=False):

    if include_split:

        components = [
            f"{item['relative_path']}|"
            f"{item['sha256']}|"
            f"{item['split']}"
            for item in records
        ]

    else:

        components = [
            f"{item['relative_path']}|{item['sha256']}"
            for item in records
        ]

    components.sort()

    content = "\n".join(components)

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


# ============================================================
# EXPORT
# ============================================================

def export_split_files(
    original_records,
    retained_records,
    split_records,
    class_split_counts,
    duplicate_rows,
    ratios,
    seed
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save individual split files.
    for split_name, split_data in split_records.items():

        save_csv(
            split_data,
            OUTPUT_DIR / f"{split_name}.csv",
            CSV_COLUMNS
        )

    # Complete manifest.
    manifest = sorted(
        [
            record
            for records in split_records.values()
            for record in records
        ],
        key=lambda item: item["relative_path"]
    )

    save_csv(
        manifest,
        OUTPUT_DIR / "split_manifest.csv",
        CSV_COLUMNS
    )

    # Original class distribution.
    source_class_counts = dict(
        Counter(
            item["class_name"]
            for item in original_records
        )
    )

    # Effective class distribution.
    effective_class_counts = dict(
        Counter(
            item["class_name"]
            for item in retained_records
        )
    )

    total_split_counts = {
        name: len(records)
        for name, records in split_records.items()
    }

    source_fingerprint = create_fingerprint(
        original_records
    )

    effective_fingerprint = create_fingerprint(
        retained_records
    )

    split_fingerprint = create_fingerprint(
        manifest,
        include_split=True
    )

    metadata = {

        "project": "Maize Leaf Classification",

        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "dataset_path": str(DATASET_DIR),

        "num_classes": NUM_CLASSES,

        "class_names": list(CLASS_NAMES),

        "class_to_idx": CLASS_TO_IDX,

        "original_image_count": len(original_records),

        "effective_image_count": len(retained_records),

        "excluded_duplicate_copies": len(duplicate_rows),

        "duplicate_groups": len({
            item["sha256"]
            for item in duplicate_rows
        }),

        "duplicate_policy": (
            "SHA-256 exact duplicate detection. "
            "Same-class duplicates keep the lexicographically "
            "first relative path. Extra copies excluded "
            "without modifying original files."
        ),

        "split_unit": "image",

        "grouping_method": (
            "Stratified random image-level split. "
            "Plant/session identifiers unavailable."
        ),

        "random_seed": seed,

        "split_ratios": ratios,

        "source_class_counts": source_class_counts,

        "effective_class_counts": effective_class_counts,

        "class_split_counts": class_split_counts,

        "total_split_counts": total_split_counts,

        "source_dataset_fingerprint_sha256": (
            source_fingerprint
        ),

        "effective_dataset_fingerprint_sha256": (
            effective_fingerprint
        ),

        "split_assignment_fingerprint_sha256": (
            split_fingerprint
        ),

        "image_integrity_verified": True,

        "all_retained_images_assigned": True,

        "exact_duplicates_removed_from_split": True,

        "cross_class_duplicate_conflicts": 0,

    }

    with open(
        OUTPUT_DIR / "split_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    return metadata


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    original_records,
    split_records,
    class_split_counts,
    duplicate_rows
):

    print("\n" + "=" * 95)

    print("FINAL DATASET SPLIT SUMMARY")

    print("=" * 95)

    print(
        f"{'Class':<40}"
        f"{'Train':>10}"
        f"{'Validation':>14}"
        f"{'Test':>10}"
        f"{'Total':>10}"
    )

    print("-" * 95)

    for class_name in CLASS_NAMES:

        counts = class_split_counts[class_name]

        total = sum(counts.values())

        print(
            f"{class_name:<40}"
            f"{counts['train']:>10}"
            f"{counts['validation']:>14}"
            f"{counts['test']:>10}"
            f"{total:>10}"
        )

    print("-" * 95)

    print(
        f"{'TOTAL':<40}"
        f"{len(split_records['train']):>10}"
        f"{len(split_records['validation']):>14}"
        f"{len(split_records['test']):>10}"
        f"{sum(len(x) for x in split_records.values()):>10}"
    )

    print("=" * 95)

    print(f"\nOriginal images: {len(original_records)}")

    print(f"Duplicate copies excluded: {len(duplicate_rows)}")

    print(
        "Effective images: "
        f"{sum(len(x) for x in split_records.values())}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create fixed stratified splits "
            "for Maize Leaf Classification."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_RATIOS["train"]
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_RATIOS["validation"]
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_RATIOS["test"]
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing split files intentionally."
    )

    args = parser.parse_args()

    ratios = {

        "train": args.train_ratio,

        "validation": args.val_ratio,

        "test": args.test_ratio,

    }

    if any(value <= 0 for value in ratios.values()):

        parser.error("All ratios must be positive.")

    if not math.isclose(
        sum(ratios.values()),
        1.0,
        abs_tol=1e-8
    ):

        parser.error(
            "All split ratios must sum to 1.0."
        )

    if args.seed < 0:

        parser.error("Seed must be non-negative.")

    print("\n" + "=" * 65)

    print("MAIZE LEAF CENTRAL SPLIT GENERATOR")

    print("=" * 65)

    print(f"\nProject: {PROJECT_ROOT}")

    print(f"Dataset: {DATASET_DIR}")

    print(f"Classes: {NUM_CLASSES}")

    print(f"Seed: {args.seed}")

    print(f"Ratios: {ratios}")

    # Prevent accidental replacement of fixed splits.
    existing_files = [
        OUTPUT_DIR / filename
        for filename in CORE_SPLIT_FILES
        if (OUTPUT_DIR / filename).exists()
    ]

    if existing_files and not args.force:

        raise FileExistsError(
            "\nExisting split files detected!\n"
            "The fixed dataset split is protected.\n"
            "Use --force only when intentionally "
            "creating a new research split."
        )

    # 1. Scan original dataset.
    original_records = scan_dataset()

    # 2. Detect and handle exact duplicates.
    retained_records, duplicate_rows = (
        deduplicate_exact_images(original_records)
    )

    # 3. Create stratified splits.
    split_records, class_split_counts = (
        create_stratified_split(
            retained_records,
            ratios,
            args.seed
        )
    )

    # 4. Validate assignments.
    validate_generated_splits(
        split_records,
        retained_records
    )

    # 5. Export all files.
    metadata = export_split_files(
        original_records,
        retained_records,
        split_records,
        class_split_counts,
        duplicate_rows,
        ratios,
        args.seed
    )

    # 6. Display summary.
    print_summary(
        original_records,
        split_records,
        class_split_counts,
        duplicate_rows
    )

    print("\nSPLIT FILES CREATED SUCCESSFULLY!")

    for filename in [
        "split_manifest.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
        "split_metadata.json",
        "exact_duplicates.csv",
    ]:

        print(f"  {OUTPUT_DIR / filename}")

    print(
        "\nSplit fingerprint: "
        f"{metadata['split_assignment_fingerprint_sha256']}"
    )

    print("\nCENTRALIZED DATASET SPLIT: PASSED")

    print("Original Dataset: UNCHANGED")

    print("All seven models must use these exact manifests.")

    print("=" * 65)


if __name__ == "__main__":

    main()
