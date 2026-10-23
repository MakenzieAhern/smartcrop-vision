"""Plot saved evaluation arrays without dropping unrepresented predictions."""

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from .labels import DEFAULT_LABELS, PROJECT_ROOT, load_labels
    from .plot_confusion import plot_cm
else:
    from labels import DEFAULT_LABELS, PROJECT_ROOT, load_labels
    from plot_confusion import plot_cm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "figs" / "confusion_matrix.png")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--counts", action="store_true", help="Show counts instead of row-normalized values.")
    args = parser.parse_args(argv)
    y_true = np.load(args.report_dir / "mini_eval_y_true.npy", allow_pickle=False)
    y_pred = np.load(args.report_dir / "mini_eval_y_pred.npy", allow_pickle=False)
    plot_cm(y_true, y_pred, normalize=not args.counts, out=args.output, labels=load_labels(args.labels))


if __name__ == "__main__":
    main()
