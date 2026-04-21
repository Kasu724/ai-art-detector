"""Experiment comparison utilities."""

from __future__ import annotations

from pathlib import Path

from ai_art_detector.utils.io import read_json, write_json


def compare_evaluation_runs(metrics_paths: list[str | Path], output_path: str | Path) -> Path:
    rows = []
    for metrics_path in metrics_paths:
        payload = read_json(Path(metrics_path))
        metrics = payload["metrics"]
        rows.append(
            {
                "metrics_path": str(metrics_path),
                "checkpoint_path": payload.get("checkpoint_path"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "ece": metrics.get("ece"),
                "threshold": metrics.get("threshold"),
                "calibrated": metrics.get("calibrated"),
            }
        )
    output = Path(output_path)
    write_json({"experiments": rows}, output)
    markdown_path = output.with_suffix(".md")
    markdown_lines = [
        "| Metrics File | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE | Threshold | Calibrated |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        markdown_lines.append(
            "| {metrics_path} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {roc_auc_display} | {ece:.4f} | {threshold:.3f} | {calibrated} |".format(
                metrics_path=row["metrics_path"],
                accuracy=row["accuracy"],
                precision=row["precision"],
                recall=row["recall"],
                f1=row["f1"],
                roc_auc_display=f"{row['roc_auc']:.4f}" if row["roc_auc"] is not None else "n/a",
                ece=row["ece"],
                threshold=row["threshold"],
                calibrated="yes" if row["calibrated"] else "no",
            )
        )
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    return output
