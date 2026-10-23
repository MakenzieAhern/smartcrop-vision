"""Load the explicit class order saved alongside the trained classifier."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "class_indices.json"


def load_labels(path=DEFAULT_LABELS):
    """Return labels by output index, rejecting ambiguous or incomplete maps."""
    with Path(path).open(encoding="utf-8") as handle:
        label_to_idx = json.load(handle)
    if not isinstance(label_to_idx, dict) or not label_to_idx:
        raise ValueError("Label map must be a nonempty JSON object.")
    if any(not isinstance(name, str) or not name for name in label_to_idx):
        raise ValueError("Label names must be nonempty strings.")
    indices = list(label_to_idx.values())
    if any(type(index) is not int for index in indices):
        raise ValueError("Label indices must be integers.")
    if sorted(indices) != list(range(len(indices))):
        raise ValueError("Label indices must be unique and contiguous, starting at 0.")
    return [name for name, _ in sorted(label_to_idx.items(), key=lambda item: item[1])]


if __name__ == "__main__":
    print(load_labels())
