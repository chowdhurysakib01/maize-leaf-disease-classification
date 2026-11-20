
"""
Maize Leaf Classification
Fixed Dataset Split Verification

Verifies:
1. Train/validation/test manifests.
2. Split manifest consistency.
3. Dataset fingerprint.
4. Split assignment fingerprint.
5. Image integrity using SHA-256.
6. Duplicate exclusion audit.
7. Seven-class consistency.
8. Dataset counts and metadata.

Does not modify any dataset or split file.
"""

import csv
import hashlib
import json

from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

from src.data.class_names import (
    PROJECT_ROOT,
    DATASET_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    NUM_CLASSES,
)


# ============================================================
# PATHS
# ============================================================

SPLIT_DIR = PROJECT_ROOT / "data_splits"

TEST_OUTPUT_DIR = PROJECT_ROOT / "Pipeline_Test"

MANIFEST_FILE = SPLIT_DIR / "split_manifest.csv"

SPLIT_FILES = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv",
}

METADATA_FILE = SPLIT_DIR / "split_metadata.json"

DUPLICATE_FILE = SPLIT_DIR / "exact_duplicates.csv"

EXPECTED_COLUMNS = [
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


# ============================================================
# GENERAL UTILITIES
# ============================================================

def require(condition, message):

    if not condition:

        raise RuntimeError(message)


def load_json(path):

    require(
        path.exists(),
        f"Required file missing: {path}"
    )

    with open(path, "r", encoding="utf-8") as file:

        return json.load(file)


def load_csv(path, expected_columns):

    require(
        path.exists(),
        f"Required CSV missing: {path}"
    )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        missing_columns = (
            set(expected_columns) -
            set(reader.fieldnames or [])
        )

        require(
            not missing_columns,
            f"Missing columns in {path.name}: "
            f"{sorted(missing_columns)}"
        )

        return list(reader)


def canonical_record(row):

    return {

        "relative_path": row["relative_path"],

        "class_name": row["class_name"],

        "class_index": int(row["class_index"]),

        "group_id": row.get("group_id", "") or "",

        "split": row["split"],

        "width": int(row["width"]),

        "height": int(row["height"]),

        "sha256": row["sha256"],

    }


# ============================================================
# HASH UTILITIES
# ============================================================

def calculate_sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            digest.update(chunk)

    return digest.hexdigest()


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
# PATH VALIDATION
# ============================================================

def resolve_project_image(relative_path):

    path = (PROJECT_ROOT / relative_path).resolve()

    require(
        path.is_relative_to(PROJECT_ROOT.resolve()),
        f"Invalid path outside project: {relative_path}"
    )

    require(
        path.is_file(),
        f"Image missing: {path}"
    )

    return path


# ============================================================
# METADATA VALIDATION
# ============================================================

def verify_class_metadata(metadata):

    require(
        metadata["num_classes"] == NUM_CLASSES,
        "Number of classes mismatch."
    )

    require(
        metadata["class_names"] == list(CLASS_NAMES),
        "Class name configuration mismatch."
    )

    require(
        metadata["class_to_idx"] == CLASS_TO_IDX,
        "Class index mapping mismatch."
    )

    require(
        metadata["split_unit"] == "image",
        "Unexpected split unit."
    )

    require(
        metadata["random_seed"] == 42,
        "Unexpected random seed."
    )

    ratios = metadata["split_ratios"]

    require(
        abs(sum(ratios.values()) - 1.0) < 1e-8,
        "Split ratios do not sum to 1."
    )

    print("[PASS] Seven-class metadata")

    print("[PASS] Class index mapping")

    print("[PASS] Split configuration")


# ============================================================
# SPLIT MANIFEST VALIDATION
# ============================================================

def load_and_verify_splits():

    print("\nChecking individual split files...")

    split_records = {}

    for split_name, path in SPLIT_FILES.items():

        raw_records = load_csv(
            path,
            EXPECTED_COLUMNS
        )

        records = [
            canonical_record(row)
            for row in raw_records
        ]

        require(
            len(records) > 0,
            f"{split_name} split is empty."
        )

        for record in records:

            require(
                record["split"] == split_name,
                f"Incorrect split assignment: "
                f"{record['relative_path']}"
            )

            require(
                record["class_name"] in CLASS_TO_IDX,
                f"Unknown class: {record['class_name']}"
            )

            require(
                record["class_index"] ==
                CLASS_TO_IDX[record["class_name"]],
                f"Incorrect class index: "
                f"{record['relative_path']}"
            )

            require(
                len(record["sha256"]) == 64,
                f"Invalid SHA-256: "
                f"{record['relative_path']}"
            )

        split_records[split_name] = records

        print(
            f"[PASS] {split_name.upper()}: "
            f"{len(records)} images"
        )

    return split_records


# ============================================================
# OVERLAP AND CLASS VALIDATION
# ============================================================

def verify_overlap_and_classes(split_records):

    print("\nChecking image overlap...")

    all_paths = []

    all_hashes = []

    class_split_counts = {}

    for split_name, records in split_records.items():

        paths = [
            record["relative_path"]
            for record in records
        ]

        hashes = [
            record["sha256"]
            for record in records
        ]

        all_paths.extend(paths)

        all_hashes.extend(hashes)

        counts = Counter(
            record["class_name"]
            for record in records
        )

        missing_classes = (
            set(CLASS_NAMES) - set(counts)
        )

        require(
            not missing_classes,
            f"Missing classes in {split_name}: "
            f"{sorted(missing_classes)}"
        )

        class_split_counts[split_name] = dict(counts)

    require(
        len(all_paths) == len(set(all_paths)),
        "Image path overlap detected between splits."
    )

    require(
        len(all_hashes) == len(set(all_hashes)),
        "Exact duplicate hash detected between splits."
    )

    print("[PASS] No image path overlap")

    print("[PASS] No exact duplicate across splits")

    print("[PASS] All seven classes present")

    return class_split_counts


# ============================================================
# MASTER MANIFEST VALIDATION
# ============================================================

def verify_master_manifest(split_records):

    print("\nChecking master manifest...")

    raw_manifest = load_csv(
        MANIFEST_FILE,
        EXPECTED_COLUMNS
    )

    manifest = [
        canonical_record(row)
        for row in raw_manifest
    ]

    combined = [

        record

        for records in split_records.values()

        for record in records

    ]

    expected = sorted(
        combined,
        key=lambda item: item["relative_path"]
    )

    actual = sorted(
        manifest,
        key=lambda item: item["relative_path"]
    )

    require(
        actual == expected,
        "Master manifest does not match individual split files."
    )

    print("[PASS] Master manifest consistency")

    return manifest


# ============================================================
# DUPLICATE AUDIT VALIDATION
# ============================================================

def verify_duplicate_audit(manifest):

    print("\nChecking duplicate audit...")

    duplicate_rows = load_csv(
        DUPLICATE_FILE,
        DUPLICATE_COLUMNS
    )

    effective_by_path = {
        record["relative_path"]: record
        for record in manifest
    }

    excluded_paths = []

    source_records = list(manifest)

    duplicate_hashes = set()

    for row in duplicate_rows:

        require(
            row["action"] == "excluded_from_split",
            "Unexpected duplicate action detected."
        )

        kept_path = row["kept_path"]

        excluded_path = row["excluded_path"]

        require(
            kept_path in effective_by_path,
            f"Duplicate keeper missing: {kept_path}"
        )

        require(
            excluded_path not in effective_by_path,
            f"Excluded duplicate appears in split: {excluded_path}"
        )

        require(
            row["class_name"] ==
            effective_by_path[kept_path]["class_name"],
            "Duplicate class mismatch."
        )

        require(
            row["sha256"] ==
            effective_by_path[kept_path]["sha256"],
            "Duplicate keeper hash mismatch."
        )

        require(
            excluded_path not in excluded_paths,
            f"Duplicate exclusion repeated: {excluded_path}"
        )

        duplicate_path = resolve_project_image(
            excluded_path
        )

        actual_hash = calculate_sha256(
            duplicate_path
        )

        require(
            actual_hash == row["sha256"],
            f"Excluded duplicate file changed: {excluded_path}"
        )

        excluded_paths.append(excluded_path)

        duplicate_hashes.add(row["sha256"])

        source_records.append({

            "relative_path": excluded_path,

            "sha256": row["sha256"],

        })

    require(
        len(excluded_paths) == len(set(excluded_paths)),
        "Repeated excluded image paths."
    )

    require(
        len({item["sha256"] for item in manifest}) ==
        len(manifest),
        "Effective dataset contains duplicate hashes."
    )

    print(
        f"[PASS] Duplicate groups: {len(duplicate_hashes)}"
    )

    print(
        f"[PASS] Excluded copies: {len(excluded_paths)}"
    )

    print("[PASS] Original duplicate files preserved")

    return source_records, len(excluded_paths), len(duplicate_hashes)


# ============================================================
# IMAGE INTEGRITY VALIDATION
# ============================================================

def verify_image_integrity(manifest):

    print("\nRechecking all retained images...")

    checked = 0

    for record in manifest:

        path = resolve_project_image(
            record["relative_path"]
        )

        actual_hash = calculate_sha256(path)

        require(
            actual_hash == record["sha256"],
            f"Image hash mismatch: {record['relative_path']}"
        )

        checked += 1

    print(
        f"[PASS] SHA-256 integrity: {checked} images"
    )


# ============================================================
# FINGERPRINT VALIDATION
# ============================================================

def verify_fingerprints(
    metadata,
    manifest,
    source_records
):

    print("\nVerifying dataset fingerprints...")

    effective_fingerprint = create_fingerprint(
        manifest
    )

    source_fingerprint = create_fingerprint(
        source_records
    )

    split_fingerprint = create_fingerprint(
        manifest,
        include_split=True
    )

    require(
        effective_fingerprint ==
        metadata["effective_dataset_fingerprint_sha256"],
        "Effective dataset fingerprint mismatch."
    )

    require(
        source_fingerprint ==
        metadata["source_dataset_fingerprint_sha256"],
        "Source dataset fingerprint mismatch."
    )

    require(
        split_fingerprint ==
        metadata["split_assignment_fingerprint_sha256"],
        "Split assignment fingerprint mismatch."
    )

    print("[PASS] Effective dataset fingerprint")

    print("[PASS] Source dataset fingerprint")

    print("[PASS] Split assignment fingerprint")

    return {
        "source": source_fingerprint,
        "effective": effective_fingerprint,
        "split": split_fingerprint,
    }


# ============================================================
# COUNT VALIDATION
# ============================================================

def verify_counts(
    metadata,
    manifest,
    split_records,
    duplicate_count,
    class_split_counts
):

    print("\nVerifying image counts...")

    effective_count = len(manifest)

    original_count = effective_count + duplicate_count

    require(
        effective_count == metadata["effective_image_count"],
        "Effective image count mismatch."
    )

    require(
        original_count == metadata["original_image_count"],
        "Original image count mismatch."
    )

    require(
        duplicate_count == metadata["excluded_duplicate_copies"],
        "Excluded duplicate count mismatch."
    )

    for split_name, records in split_records.items():

        require(
            len(records) ==
            metadata["total_split_counts"][split_name],
            f"{split_name} count mismatch."
        )

    for class_name in CLASS_NAMES:

        actual_total = sum(
            1 for record in manifest
            if record["class_name"] == class_name
        )

        expected_total = metadata[
            "effective_class_counts"
        ][class_name]

        require(
            actual_total == expected_total,
            f"Class count mismatch: {class_name}"
        )

        for split_name in split_records:

            actual_split = class_split_counts[
                split_name
            ][class_name]

            expected_split = metadata[
                "class_split_counts"
            ][class_name][split_name]

            require(
                actual_split == expected_split,
                f"Class/split count mismatch: "
                f"{class_name}, {split_name}"
            )

    print("[PASS] Original image count")

    print("[PASS] Effective image count")

    print("[PASS] Duplicate exclusion count")

    print("[PASS] All class-wise split counts")


# ============================================================
# REPORT
# ============================================================

def save_verification_report(
    metadata,
    fingerprints,
    split_records,
    duplicate_count,
    duplicate_groups
):

    TEST_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {

        "project": "Maize Leaf Classification",

        "verification_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "status": "PASSED",

        "num_classes": NUM_CLASSES,

        "class_names": list(CLASS_NAMES),

        "original_images": metadata["original_image_count"],

        "effective_images": metadata["effective_image_count"],

        "excluded_duplicate_copies": duplicate_count,

        "duplicate_groups": duplicate_groups,

        "split_counts": {

            name: len(records)

            for name, records in split_records.items()

        },

        "seed": metadata["random_seed"],

        "fingerprints": fingerprints,

        "same_split_required_for_all_models": True,

        "group_level_leakage_verified": False,

        "image_level_overlap_verified": True,

        "exact_duplicate_integrity_verified": True,

    }

    output_path = (
        TEST_OUTPUT_DIR /
        "split_verification_report.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nReport saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print("MAIZE LEAF - DATASET SPLIT VERIFICATION")

    print("=" * 70)

    print(f"\nProject: {PROJECT_ROOT}")

    print(f"Dataset: {DATASET_DIR}")

    metadata = load_json(METADATA_FILE)

    verify_class_metadata(metadata)

    split_records = load_and_verify_splits()

    class_split_counts = verify_overlap_and_classes(
        split_records
    )

    manifest = verify_master_manifest(
        split_records
    )

    source_records, duplicate_count, duplicate_groups = (
        verify_duplicate_audit(manifest)
    )

    verify_image_integrity(manifest)

    fingerprints = verify_fingerprints(
        metadata,
        manifest,
        source_records
    )

    verify_counts(
        metadata,
        manifest,
        split_records,
        duplicate_count,
        class_split_counts
    )

    save_verification_report(
        metadata,
        fingerprints,
        split_records,
        duplicate_count,
        duplicate_groups
    )

    print("\n" + "=" * 70)

    print("ALL DATASET VERIFICATION CHECKS PASSED")

    print("=" * 70)

    print("\nFIXED DATASET SPLIT: VERIFIED")

    print("ALL SEVEN MODELS: SAME SPLIT REQUIRED")

    print("ORIGINAL DATASET: UNCHANGED")

    print("=" * 70)


if __name__ == "__main__":

    main()
