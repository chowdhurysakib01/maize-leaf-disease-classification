"""Collect fine-tuning artifacts into the completed evaluation run and plot history."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def copy_training_artifacts(training_dir: Path, final_dir: Path) -> Path:
    fine_dir = final_dir / "fine_tuning"
    fine_dir.mkdir(parents=True, exist_ok=True)

    required = [training_dir / "history.csv", training_dir / "training_summary.json", training_dir / "checkpoints" / "best_model.pth"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing fine-tuning artifacts:\n" + "\n".join(missing))

    shutil.copy2(training_dir / "history.csv", fine_dir / "history.csv")
    shutil.copy2(training_dir / "training_summary.json", fine_dir / "training_summary.json")
    shutil.copy2(training_dir / "history.csv", final_dir / "fine_tuning_history.csv")

    checkpoint_dst = fine_dir / "checkpoints"
    if checkpoint_dst.exists():
        shutil.rmtree(checkpoint_dst)
    shutil.copytree(training_dir / "checkpoints", checkpoint_dst)

    # Keep the exact source checkpoint and last-stage recovery checkpoint together.
    return fine_dir


def make_training_plots(history_path: Path, summary: dict[str, Any], fine_dir: Path) -> dict[str, str]:
    df = pd.read_csv(history_path)
    if df.empty:
        raise ValueError("Fine-tuning history is empty; cannot create training plots.")

    curves_dir = fine_dir / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    # Loss curve with the pre-fine-tuning validation baseline.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["train_loss"], marker="o", linewidth=1.8, label="Training loss")
    ax.plot(df["epoch"], df["validation_loss"], marker="o", linewidth=1.8, label="Validation loss")
    baseline_loss = summary.get("reproduced_baseline_val_loss")
    if baseline_loss is not None:
        ax.axhline(float(baseline_loss), linestyle="--", linewidth=1.4, label="Base DMNet validation loss")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Loss")
    ax.set_title("RiceLeaf-DMNet Fine-Tuning: Training vs Validation Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    loss_path = curves_dir / "training_validation_loss.png"
    fig.savefig(loss_path, dpi=220)
    plt.close(fig)
    outputs["training_validation_loss"] = str(loss_path)

    # Validation accuracy and macro-F1, with the starting checkpoint baseline.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["validation_accuracy"], marker="o", linewidth=1.8, label="Validation accuracy")
    ax.plot(df["epoch"], df["validation_macro_f1"], marker="o", linewidth=1.8, label="Validation macro-F1")
    baseline_acc = summary.get("reproduced_baseline_val_accuracy")
    baseline_f1 = summary.get("reproduced_baseline_val_macro_f1")
    if baseline_acc is not None:
        ax.axhline(float(baseline_acc), linestyle="--", linewidth=1.2, label="Base DMNet validation accuracy")
    if baseline_f1 is not None:
        ax.axhline(float(baseline_f1), linestyle=":", linewidth=1.5, label="Base DMNet validation macro-F1")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("RiceLeaf-DMNet Fine-Tuning: Validation Metrics")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    metric_path = curves_dir / "validation_metrics.png"
    fig.savefig(metric_path, dpi=220)
    plt.close(fig)
    outputs["validation_metrics"] = str(metric_path)

    # Learning-rate chart: each epoch may contain multiple parameter-group rates.
    if "learning_rates" in df.columns:
        import ast
        parsed = []
        for raw in df["learning_rates"].fillna("[]"):
            try:
                values = json.loads(raw) if isinstance(raw, str) else []
                parsed.append(max([float(v) for v in values], default=float("nan")))
            except Exception:
                parsed.append(float("nan"))
        if any(pd.notna(v) for v in parsed):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["epoch"], parsed, marker="o", linewidth=1.6)
            ax.set_xlabel("Fine-tuning epoch")
            ax.set_ylabel("Maximum parameter-group learning rate")
            ax.set_yscale("log")
            ax.set_title("Fine-Tuning Learning Rate")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            lr_path = curves_dir / "learning_rate.png"
            fig.savefig(lr_path, dpi=220)
            plt.close(fig)
            outputs["learning_rate"] = str(lr_path)

    # Record the actual best-epoch row and last-epoch loss gap for transparent review.
    best_epoch = int(summary.get("best_epoch", 0) or 0)
    best_row = df.loc[df["epoch"] == best_epoch]
    if best_row.empty:
        best_row = df.loc[[df["validation_macro_f1"].idxmax()]]
    best_record = best_row.iloc[0].to_dict()
    last_record = df.iloc[-1].to_dict()
    diagnostics = {
        "epochs_completed": int(len(df)),
        "best_epoch_from_training_summary": best_epoch,
        "best_epoch_metrics": {k: (float(v) if pd.notna(v) and isinstance(v, (int, float)) else v) for k, v in best_record.items()},
        "last_epoch_metrics": {k: (float(v) if pd.notna(v) and isinstance(v, (int, float)) else v) for k, v in last_record.items()},
        "baseline_validation": {
            "loss": summary.get("reproduced_baseline_val_loss"),
            "accuracy": summary.get("reproduced_baseline_val_accuracy"),
            "macro_f1": summary.get("reproduced_baseline_val_macro_f1"),
        },
        "best_finetuned_validation_macro_f1": summary.get("best_finetuned_val_macro_f1"),
        "absolute_macro_f1_change": summary.get("absolute_macro_f1_change"),
        "last_epoch_train_validation_loss_gap": float(last_record["validation_loss"] - last_record["train_loss"]),
        "interpretation_note": "A positive train-validation loss gap alone does not prove overfitting; inspect the full curves and validation metrics together.",
        "plots": outputs,
    }
    write_json(fine_dir / "fine_tuning_diagnostics.json", diagnostics)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect fine-tuning and evaluation artifacts into one final run directory.")
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--final-run-dir", type=Path, required=True)
    args = parser.parse_args()

    training_dir = args.training_dir.expanduser().resolve()
    final_dir = args.final_run_dir.expanduser().resolve()
    if not training_dir.is_dir():
        raise NotADirectoryError(training_dir)
    if not final_dir.is_dir():
        raise NotADirectoryError(final_dir)

    evaluation_summary_path = final_dir / "complete_evaluation_summary.json"
    if not evaluation_summary_path.is_file():
        raise FileNotFoundError("Evaluation summary is missing. The evaluation stage may not have completed.")

    fine_dir = copy_training_artifacts(training_dir, final_dir)
    training_summary = read_json(fine_dir / "training_summary.json")
    evaluation_summary = read_json(evaluation_summary_path)
    plots = make_training_plots(fine_dir / "history.csv", training_summary, fine_dir)

    # Rewrite references from the temporary staging directory to the final
    # consolidated folder before the staging directory is removed by PowerShell.
    final_checkpoint = fine_dir / "checkpoints" / "best_model.pth"
    training_summary["best_checkpoint"] = str(final_checkpoint)
    training_summary["history_csv"] = str(fine_dir / "history.csv")
    write_json(fine_dir / "training_summary.json", training_summary)

    best_info_path = fine_dir / "checkpoints" / "best_model_info.json"
    if best_info_path.is_file():
        best_info = read_json(best_info_path)
        best_info["checkpoint"] = str(final_checkpoint)
        write_json(best_info_path, best_info)

    evaluation_summary["checkpoint"] = str(final_checkpoint)
    write_json(evaluation_summary_path, evaluation_summary)

    evaluation_metadata_path = final_dir / "evaluation_metadata.json"
    if evaluation_metadata_path.is_file():
        evaluation_metadata = read_json(evaluation_metadata_path)
        evaluation_metadata["checkpoint_used"] = str(final_checkpoint)
        write_json(evaluation_metadata_path, evaluation_metadata)

    complete_summary = {
        "pipeline_status": "complete",
        "model": "RiceLeaf-DMNet Fine-Tuned",
        "training": training_summary,
        "evaluation": evaluation_summary,
        "training_artifacts_dir": str(fine_dir),
        "training_history_csv": str(fine_dir / "history.csv"),
        "training_plots": plots,
        "test_partition_accessed": False,
    }
    write_json(final_dir / "complete_run_summary.json", complete_summary)

    run_summary_path = final_dir / "run_summary.json"
    if run_summary_path.is_file():
        run_summary = read_json(run_summary_path)
        run_summary["status"] = "complete_training_and_evaluation"
        run_summary["source_checkpoint"] = str(final_checkpoint)
        run_summary["fine_tuning_summary"] = str(fine_dir / "training_summary.json")
        run_summary["complete_run_summary"] = str(final_dir / "complete_run_summary.json")
        run_summary["test_partition_accessed"] = False
        write_json(run_summary_path, run_summary)

    print("[PASS] Fine-tuning checkpoints and history copied into final run folder")
    print("[PASS] Training/validation loss and validation-metric curves generated")
    print("[PASS] Complete training + evaluation summary generated")
    print("Final result directory:", final_dir)


if __name__ == "__main__":
    main()
