
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = "qwen2.5:3b"

XAI_METHODS = (
    "gradcam",
    "gradcam_plus_plus",
    "lime",
)

EXCLUDED_DIRS = {
    "checkpoints",
    "arrays",
    "__pycache__",
    "shap_gradient",
}

IMPORTANT_JSON = {
    "experiment_metadata.json",
    "model_architecture.json",
    "run_summary.json",
    "training_config.json",
    "split_metadata_snapshot.json",
    "run_metadata.json",
    "training_history.json",
    "metrics.json",
    "final_validation_metrics.json",
    "predictions_metadata.json",
    "efficiency_benchmark.json",
    "error_analysis_summary.json",
    "gradcam_summary.json",
    "gradcam_plus_plus_summary.json",
    "lime_summary.json",
}

IMPORTANT_CSV = {
    "training_history.csv",
    "classification_report.csv",
    "confusion_matrix.csv",
    "normalized_confusion_matrix.csv",
    "roc_auc_per_class.csv",
    "pr_auc_per_class.csv",
    "per_class_error_analysis.csv",
    "confusion_pairs.csv",
    "high_confidence_errors.csv",
    "confidence_bins.csv",
    "class_summary.csv",
    "gradcam_results.csv",
    "gradcam_plus_plus_results.csv",
    "lime_results.csv",
    "lime_features.csv",
}

IMPORTANT_TXT = {
    "model_architecture.txt",
    "training_log.txt",
    "gradcam_report.txt",
    "gradcam_plus_plus_report.txt",
    "lime_report.txt",
    "efficiency_benchmark_report.txt",
    "error_analysis_report.txt",
}

MAX_FILE_CHARS = 10000
MAX_EVIDENCE_CHARS = 42000
MAX_CSV_ROWS = 40


# ============================================================
# PRINTING
# ============================================================

def section(title: str) -> None:

    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def success(message: str) -> None:

    print(f"[PASS] {message}")


def warning(message: str) -> None:

    print(f"[WARNING] {message}")


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_run_directory(run_dir: Path) -> Path:

    run_dir = run_dir.expanduser().resolve()

    if not run_dir.exists():

        raise FileNotFoundError(
            f"Run directory not found: {run_dir}"
        )

    if not run_dir.is_dir():

        raise NotADirectoryError(str(run_dir))

    required = [
        run_dir / "training_config.json",
        run_dir / "training_history",
        run_dir / "evaluation",
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Required experiment outputs are missing:\n"
            + "\n".join(missing)
        )

    return run_dir


# ============================================================
# SOURCE DISCOVERY
# ============================================================

def is_excluded(path: Path) -> bool:

    return any(
        part.lower() in EXCLUDED_DIRS
        for part in path.parts
    )


def discover_sources(run_dir: Path) -> list[Path]:

    selected = []

    for path in run_dir.rglob("*"):

        if not path.is_file():
            continue

        if is_excluded(path):
            continue

        if path.suffix.lower() == ".json":

            if path.name in IMPORTANT_JSON:

                selected.append(path)

            elif (
                "xai" in str(path).lower()
                and path.name.endswith("_summary.json")
            ):

                selected.append(path)

        elif path.suffix.lower() == ".csv":

            if path.name in IMPORTANT_CSV:

                selected.append(path)

        elif path.suffix.lower() == ".txt":

            if path.name in IMPORTANT_TXT:

                selected.append(path)

    return sorted(set(selected))


# ============================================================
# JSON READER
# ============================================================

def read_json(path: Path) -> Any:

    with path.open(
        "r",
        encoding="utf-8-sig"
    ) as file:

        return json.load(file)


# ============================================================
# CSV READER
# ============================================================

def read_csv_summary(path: Path) -> dict:

    result = {
        "columns": [],
        "total_rows": 0,
        "sampled_rows": [],
    }

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        result["columns"] = reader.fieldnames or []

        for row in reader:

            result["total_rows"] += 1

            if len(result["sampled_rows"]) < MAX_CSV_ROWS:

                result["sampled_rows"].append(row)

    result["rows_included"] = len(
        result["sampled_rows"]
    )

    return result


# ============================================================
# SOURCE LOADING
# ============================================================

def load_source(path: Path, run_dir: Path) -> dict | None:

    relative = str(
        path.relative_to(run_dir)
    )

    try:

        if path.suffix.lower() == ".json":

            content = read_json(path)

            text_content = json.dumps(
                content,
                ensure_ascii=False,
                indent=2,
                default=str
            )

        elif path.suffix.lower() == ".csv":

            content = read_csv_summary(path)

            text_content = json.dumps(
                content,
                ensure_ascii=False,
                indent=2
            )

        elif path.suffix.lower() == ".txt":

            text_content = path.read_text(
                encoding="utf-8-sig",
                errors="replace"
            )

            content = text_content

        else:

            return None

        original_length = len(text_content)

        if original_length > MAX_FILE_CHARS:

            text_content = (
                text_content[:MAX_FILE_CHARS]
                + "\n[CONTENT TRUNCATED]"
            )

        return {
            "source": relative,
            "file_type": path.suffix.lower(),
            "content": text_content,
            "original_characters": original_length,
        }

    except Exception as error:

        warning(
            f"Could not read {relative}: {error}"
        )

        return None


def build_evidence_bundle(run_dir: Path) -> dict:

    section("COLLECTING EXPERIMENTAL EVIDENCE")

    sources = discover_sources(run_dir)

    print("Discovered files:", len(sources))

    evidence = []

    for path in sources:

        item = load_source(path, run_dir)

        if item is not None:

            evidence.append(item)

            success(item["source"])

    if not evidence:

        raise RuntimeError(
            "No usable experimental files were found."
        )

    total_chars = sum(
        len(item["content"])
        for item in evidence
    )

    if total_chars > MAX_EVIDENCE_CHARS:

        warning(
            "Evidence exceeds context budget. "
            "Less essential sources will be excluded."
        )

        priority = [
            "training_config.json",
            "training_history.json",
            "run_summary.json",
            "metrics.json",
            "final_validation_metrics.json",
            "classification_report.csv",
            "confusion_matrix.csv",
            "efficiency_benchmark.json",
            "error_analysis_summary.json",
            "per_class_error_analysis.csv",
            "confusion_pairs.csv",
            "gradcam_summary.json",
            "gradcam_plus_plus_summary.json",
            "lime_summary.json",
        ]

        def rank(item):

            name = Path(item["source"]).name

            try:
                return priority.index(name)
            except ValueError:
                return len(priority)

        evidence.sort(key=rank)

        reduced = []
        used = 0

        for item in evidence:

            size = len(item["content"])

            if used + size > MAX_EVIDENCE_CHARS:

                continue

            reduced.append(item)

            used += size

        evidence = reduced

    bundle = {
        "project": "Maize Leaf Classification",
        "model": run_dir.parent.name,
        "run_name": run_dir.name,
        "generated_at": datetime.now().isoformat(),
        "evidence_sources": evidence,
    }

    print(
        "\nEvidence files included:",
        len(evidence)
    )

    print(
        "Evidence characters:",
        sum(
            len(x["content"])
            for x in evidence
        )
    )

    return bundle


# ============================================================
# PROMPT GENERATION
# ============================================================

def build_prompt(
    bundle: dict,
    language: str
) -> str:

    evidence = json.dumps(
        bundle,
        ensure_ascii=False,
        indent=2
    )

    if language == "bangla":

        language_instruction = """
Write the entire report in professional academic Bangla.

Use standard scientific terminology. English technical terms
may be retained in parentheses when appropriate.

Keep metric names, class names, model names and numerical
values exactly as provided in the evidence.
"""

    else:

        language_instruction = """
Write the entire report in clear, formal academic English.
Use scientific terminology suitable for a research paper.
"""

    return f"""
You are a scientific research assistant specializing in
machine learning, computer vision and agricultural
plant-disease classification.

PROJECT:
Seven-class maize leaf classification.

MODEL:
{bundle['model']}

EXPERIMENT:
{bundle['run_name']}

{language_instruction}

============================================================
STRICT SCIENTIFIC INTEGRITY RULES
============================================================

1. Use ONLY information contained in the supplied evidence.

2. NEVER invent accuracy, precision, recall, F1-score,
   AUC, training time, inference speed or other values.

3. NEVER invent experimental observations.

4. Do not claim external validation unless evidence
   explicitly demonstrates it.

5. Distinguish validation results from test results.

6. Do not describe validation performance as test performance.

7. Do not claim statistical significance without a
   statistical test and its results.

8. Do not claim robustness without actual robustness tests.

9. Do not claim that XAI proves causality.

10. If information is missing, explicitly state:
    "Not available in the supplied experimental evidence."

11. Clearly distinguish measured results from interpretation.

12. Preserve all seven original class names.

13. Do not assume that an empty SHAP directory means
    SHAP experiments were completed.

14. XAI image files themselves have not been supplied
    as visual inputs. Interpret only the supplied XAI
    CSV and JSON reports, not unseen image contents.

15. Do not recommend a research novelty as an established
    contribution without experimental evidence.

============================================================
REQUIRED REPORT STRUCTURE
============================================================

# 1. Experiment Overview

Explain model, dataset, classes, training configuration
and experimental setup using available information.

# 2. Training Behaviour

Discuss epoch-wise training loss, validation loss,
training accuracy, validation accuracy, learning rate,
early stopping and training duration.

# 3. Classification Performance

Discuss available:

- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1
- Balanced Accuracy
- ROC-AUC
- PR-AUC

Use exact reported values.

# 4. Class-wise Performance

Explain the performance of all seven classes.

Identify strong and weak classes using the evidence.

# 5. Confusion and Error Analysis

Discuss the most frequently confused classes,
misclassification patterns and confidence-related errors.

# 6. Computational Efficiency

Discuss:

- Total parameters
- Trainable parameters
- Frozen parameters
- FLOPs/MACs
- Model size
- Training time
- Inference latency
- Throughput
- GPU memory

Only report values actually available.

# 7. Explainable AI Interpretation

Provide separate subsections for:

- Grad-CAM
- Grad-CAM++
- LIME

Explain what the supplied reports establish.

Discuss agreement or disagreement only when supported
by comparable evidence.

# 8. Model Strengths

Identify evidence-supported strengths.

# 9. Model Limitations

Identify evidence-supported limitations and missing tests.

# 10. Research Implications

Explain what the current findings suggest for further
maize leaf classification research.

Do not claim novelty has been established.

# 11. Missing Experimental Evidence

List important missing experiments, metrics and
validations required before publication.

# 12. Overall Conclusion

Give a balanced scientific conclusion.

============================================================
OUTPUT QUALITY
============================================================

- Use headings and subheadings.
- Use tables when they improve clarity.
- Keep numerical values consistent.
- Avoid repetitive generic statements.
- Do not exaggerate the model's performance.
- Explain limitations honestly.
- Make the report useful for manuscript preparation.

============================================================
EXPERIMENTAL EVIDENCE
============================================================

{evidence}

============================================================

Generate the complete report now.
"""


# ============================================================
# OLLAMA
# ============================================================

def check_ollama(model: str) -> None:

    if shutil.which("ollama") is None:

        raise RuntimeError(
            "Ollama executable was not found. "
            "Install Ollama or add it to PATH."
        )

    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Could not communicate with Ollama.\n"
            + result.stderr
        )

    installed = result.stdout

    if model not in installed:

        raise RuntimeError(
            f"Model {model} was not found in ollama list."
        )

    success(f"Ollama model available: {model}")


def generate_with_ollama(
    prompt: str,
    model: str,
    timeout: int
) -> str:

    start = time.perf_counter()

    result = subprocess.run(
        [
            "ollama",
            "run",
            model
        ],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout
    )

    elapsed = time.perf_counter() - start

    if result.returncode != 0:

        raise RuntimeError(
            "Ollama generation failed:\n"
            + result.stderr
        )

    response = result.stdout.strip()

    if not response:

        raise RuntimeError(
            "Ollama returned an empty response."
        )

    print(
        f"Generation time: {elapsed / 60:.2f} minutes"
    )

    return response


# ============================================================
# REPORT SECTION EXTRACTION
# ============================================================

def extract_sections(
    report: str,
    headings: list[str]
) -> str:

    lines = report.splitlines()

    selected = []
    collecting = False

    for line in lines:

        clean = line.strip().lower()

        if clean.startswith("#"):

            if any(
                heading.lower() in clean
                for heading in headings
            ):

                collecting = True

            elif collecting:

                break

        if collecting:

            selected.append(line)

    if selected:

        return "\n".join(selected).strip()

    return report


def save_text(path: Path, content: str) -> None:

    path.write_text(
        content.strip() + "\n",
        encoding="utf-8"
    )


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_reports(
    bundle: dict,
    output_dir: Path,
    model: str,
    timeout: int
) -> None:

    section("GENERATING ENGLISH REPORT")

    english_prompt = build_prompt(
        bundle,
        "english"
    )

    english_report = generate_with_ollama(
        english_prompt,
        model,
        timeout
    )

    save_text(
        output_dir / "llm_analysis_english.md",
        english_report
    )

    success("English report saved")

    section("GENERATING BANGLA REPORT")

    bangla_prompt = build_prompt(
        bundle,
        "bangla"
    )

    bangla_report = generate_with_ollama(
        bangla_prompt,
        model,
        timeout
    )

    save_text(
        output_dir / "llm_analysis_bangla.md",
        bangla_report
    )

    success("Bangla report saved")

    # Separate English research documents.

    save_text(
        output_dir / "research_findings.md",
        extract_sections(
            english_report,
            [
                "experiment overview",
                "training behaviour",
                "classification performance",
                "class-wise performance",
                "confusion and error analysis",
                "computational efficiency",
                "research implications",
                "overall conclusion",
            ]
        )
    )

    save_text(
        output_dir / "xai_interpretation.md",
        extract_sections(
            english_report,
            ["explainable ai interpretation"]
        )
    )

    save_text(
        output_dir / "model_strengths_limitations.md",
        extract_sections(
            english_report,
            [
                "model strengths",
                "model limitations",
            ]
        )
    )

    success("Research findings extracted")
    success("XAI interpretation extracted")
    success("Strengths and limitations extracted")


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    bundle: dict,
    output_dir: Path,
    model: str
) -> None:

    metadata = {
        "project": bundle["project"],
        "model": bundle["model"],
        "run_name": bundle["run_name"],
        "ollama_model": model,
        "generated_at": datetime.now().isoformat(),
        "language_outputs": [
            "English",
            "Bangla",
        ],
        "source_files": [
            item["source"]
            for item in bundle["evidence_sources"]
        ],
        "source_file_count": len(
            bundle["evidence_sources"]
        ),
        "xai_methods_expected": list(
            XAI_METHODS
        ),
        "shap_included": False,
        "evidence_based_generation": True,
    }

    path = output_dir / "llm_analysis_metadata.json"

    path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    success("LLM metadata saved")


# ============================================================
# MAIN
# ============================================================

def run_pipeline(args) -> None:

    start_time = time.perf_counter()

    run_dir = validate_run_directory(
        Path(args.run_dir)
    )

    output_dir = run_dir / "LLM"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    section("MAIZE LEAF - LOCAL LLM REPORT GENERATOR")

    print("Project root :", PROJECT_ROOT)
    print("Run directory:", run_dir)
    print("Output       :", output_dir)
    print("LLM model    :", args.model)
    print("Language     : English + Bangla")
    print("SHAP         : Excluded")

    check_ollama(args.model)

    bundle = build_evidence_bundle(
        run_dir
    )

    evidence_path = (
        output_dir / "evidence_bundle.json"
    )

    evidence_path.write_text(
        json.dumps(
            bundle,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    success("Evidence bundle saved")

    if args.dry_run:

        section("DRY RUN COMPLETED")

        print(
            "Evidence file:",
            evidence_path
        )

        print(
            "Source files:",
            len(bundle["evidence_sources"])
        )

        return

    generate_reports(
        bundle,
        output_dir,
        args.model,
        args.timeout
    )

    save_metadata(
        bundle,
        output_dir,
        args.model
    )

    elapsed = time.perf_counter() - start_time

    log = {
        "completed": True,
        "model": args.model,
        "run_name": run_dir.name,
        "elapsed_seconds": round(elapsed, 2),
        "completed_at": datetime.now().isoformat(),
    }

    (output_dir / "llm_run_log.json").write_text(
        json.dumps(
            log,
            indent=4
        ),
        encoding="utf-8"
    )

    section("LLM REPORT GENERATION COMPLETED")

    print("Model       :", args.model)
    print("Time        :", f"{elapsed / 60:.2f} minutes")
    print("Output      :", output_dir)

    print("\nGenerated files:")

    for path in sorted(output_dir.iterdir()):

        if path.is_file():

            print(" -", path.name)

    print("\nALL LLM REPORTS GENERATED SUCCESSFULLY")


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Maize Leaf Local LLM Research "
            "Report Generator"
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed experiment run directory."
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1800
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect evidence without generating reports."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    try:

        run_pipeline(args)

    except KeyboardInterrupt:

        print("\n[INTERRUPTED] User stopped generation.")

        sys.exit(130)

    except Exception as error:

        print(
            f"\n[LLM PIPELINE ERROR] "
            f"{type(error).__name__}: {error}"
        )

        sys.exit(1)


if __name__ == "__main__":

    main()
