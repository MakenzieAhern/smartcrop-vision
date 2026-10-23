"""Render confusion matrices with every model class retained."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

if __package__:
    from .labels import PROJECT_ROOT, load_labels
else:
    from labels import PROJECT_ROOT, load_labels


def plot_cm(y_true, y_pred, normalize=True, out=None, labels=None):
    labels = load_labels() if labels is None else list(labels)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or not y_true.size:
        raise ValueError("Expected matching, nonempty one-dimensional prediction arrays.")
    for values in (y_true, y_pred):
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= len(labels)):
            raise ValueError("Prediction indices must be integers within the label map.")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    display = matrix.astype(float)
    if normalize:
        totals = display.sum(axis=1, keepdims=True)
        display = np.divide(display, totals, out=np.zeros_like(display), where=totals != 0)
    fig, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(display, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(image, ax=axis, label="Fraction of true class" if normalize else "Images")
    axis.set(
        title="Leaf disease classification — confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.setp(axis.get_xticklabels(), rotation=65, ha="right", fontsize=8)
    plt.setp(axis.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    destination = Path(out) if out is not None else PROJECT_ROOT / "figs" / "confusion_matrix.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    print(f"Saved: {destination}")
    return matrix
