from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(eval_results: dict[str, Any], output_path: str):
    """Plot and save a confusion matrix heatmap."""
    cm = np.array(eval_results["classification"]["confusion_matrix"])
    labels = eval_results["classification"]["labels"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Activity Classification Confusion Matrix")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def plot_timeline_comparison(
    eval_results: dict[str, Any], y_true: list[str], y_pred: list[str], output_path: str
):
    """Plot a side-by-side timeline of ground truth vs predicted."""
    # Create a mapping from label to integer for plotting
    labels = eval_results["classification"]["labels"]
    label_map = {lbl: i for i, lbl in enumerate(labels)}

    yt = [label_map[y] for y in y_true]
    yp = [label_map[y] for y in y_pred]

    time = np.arange(len(yt))

    plt.figure(figsize=(15, 6))
    plt.plot(time, yt, label="Ground Truth", marker=".", linestyle="", alpha=0.5)
    plt.plot(time, yp, label="Predicted", marker="x", linestyle="", alpha=0.5)

    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Time (seconds)")
    plt.title("Timeline Comparison")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
