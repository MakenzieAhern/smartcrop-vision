"""Evaluate class folders reproducibly and preserve every prediction in reports."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if __package__:
    from .labels import DEFAULT_LABELS, PROJECT_ROOT, load_labels
    from .predict_one import MODEL, load_inference_model, predict_probabilities, preprocess
else:
    from labels import DEFAULT_LABELS, PROJECT_ROOT, load_labels
    from predict_one import MODEL, load_inference_model, predict_probabilities, preprocess

MODEL_PATH = MODEL
RAND_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff"}


def collect_files(root_folder, labels, per_class=30, shuffle=True):
    """Select sorted files with a fixed seed; 0 uses all supported images.

    Unknown nonempty class folders fail explicitly: they must not disappear
    silently from evaluation totals. Images directly inside each class are used.
    """
    root = Path(root_folder)
    if not root.is_dir():
        raise ValueError(f"Evaluation folder does not exist: {root}")
    if per_class < 0:
        raise ValueError("per_class must be nonnegative; 0 selects all images.")
    known_labels = set(labels)
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith(".") and subdir.name not in known_labels:
            if any(file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS for file in subdir.iterdir()):
                raise ValueError(f"Unknown class folder: {subdir.name}; update the label map or dataset.")
    rng = random.Random(RAND_SEED)
    class_to_files = {}
    for label in labels:
        subdir = root / label
        if not subdir.is_dir():
            continue
        files = sorted(
            str(file) for file in subdir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )
        if shuffle:
            rng.shuffle(files)
        selected = files[:per_class] if per_class else files
        if selected:
            class_to_files[label] = selected
    return class_to_files, sum(map(len, class_to_files.values()))


def evaluate_folder(model, root_folder, labels, per_class=30, batch_size=32):
    """Return JSON-serializable metrics, a full confusion matrix and predictions."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    class_to_files, total = collect_files(root_folder, labels, per_class=per_class)
    if not total:
        raise ValueError(f"No valid class folders with images found under: {root_folder}")
    label_to_index = {label: index for index, label in enumerate(labels)}
    records = [(file, label_to_index[label]) for label, files in class_to_files.items() for file in files]
    y_true, y_pred, predictions = [], [], []
    for start in range(0, total, batch_size):
        batch_records = records[start:start + batch_size]
        batch = np.concatenate([preprocess(file) for file, _ in batch_records], axis=0)
        probabilities = predict_probabilities(model, batch, labels)
        for (file, true_index), probability in zip(batch_records, probabilities):
            pred_index = int(np.argmax(probability))
            y_true.append(true_index)
            y_pred.append(pred_index)
            predictions.append({
                "file": str(Path(file).relative_to(root_folder)),
                "true_index": true_index,
                "predicted_index": pred_index,
                "true_label": labels[true_index],
                "predicted_label": labels[pred_index],
                "confidence": float(probability[pred_index]),
            })
    all_indices = list(range(len(labels)))
    report = classification_report(
        y_true, y_pred, labels=all_indices, target_names=labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, labels=all_indices, target_names=labels, digits=4, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=all_indices)
    return {
        "sample_count": total,
        "class_counts": {label: len(class_to_files.get(label, [])) for label in labels},
        "labels": list(labels),
        "label_indices": all_indices,
        "represented_label_indices": sorted(set(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_all_classes": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "report_text": report_text,
        "confusion_matrix": matrix.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "predictions": predictions,
        "sampling": {"per_class": per_class, "seed": RAND_SEED, "batch_size": batch_size},
        "metric_notes": (
            "Every model class is retained in the report and confusion matrix. "
            "Macro F1 averages all classes, including classes absent from this folder. "
            "These folder-level metrics do not establish a held-out evaluation split."
        ),
    }


def save_results(results, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (output / "classification_report.txt").write_text(results["report_text"], encoding="utf-8")
    for key in ("y_true", "y_pred", "label_indices"):
        np.save(output / f"mini_eval_{key}.npy", np.asarray(results[key], dtype=np.int64))
    np.save(output / "confusion_matrix.npy", np.asarray(results["confusion_matrix"], dtype=np.int64))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, help="Root folder containing class subfolders.")
    parser.add_argument("--per_class", "--per-class", type=int, default=30, help="Max images per class (0 = all).")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to a trusted Keras checkpoint.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="JSON class-to-index map.")
    parser.add_argument("--save_preds", "--save-preds", action="store_true", help="Save reports to reports/ (legacy flag).")
    parser.add_argument("--output-dir", help="Save JSON, text report and prediction arrays to this folder.")
    args = parser.parse_args(argv)
    if args.per_class < 0 or args.batch_size < 1:
        parser.error("--per_class must be nonnegative and --batch-size must be positive")
    labels = load_labels(args.labels)
    model = load_inference_model(args.model, labels)
    results = evaluate_folder(model, args.folder, labels, args.per_class, args.batch_size)
    results["model"] = Path(args.model).name
    dataset_path = Path(args.folder)
    results["dataset"] = dataset_path.name if dataset_path.is_absolute() else str(dataset_path)
    results["model_sha256"] = file_sha256(args.model)
    results["labels_sha256"] = file_sha256(args.labels)
    results["preprocessing"] = {"image_size": [224, 224], "color_mode": "RGB", "interpolation": "nearest", "rescale": "1/255"}
    print(f"Evaluated {results['sample_count']} images; accuracy: {results['accuracy']:.4%}")
    print(results["report_text"])
    print(f"Confusion matrix: {len(labels)} x {len(labels)}; all {results['sample_count']} predictions retained.")
    if args.output_dir or args.save_preds:
        destination = args.output_dir or PROJECT_ROOT / "reports"
        save_results(results, destination)
        print(f"Saved evaluation results to: {destination}")


if __name__ == "__main__":
    main()
