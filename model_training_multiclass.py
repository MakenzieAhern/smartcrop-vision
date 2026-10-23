"""Train a reproducible MobileNetV2 leaf classifier; importing never trains.

The supplied historical checkpoints use RGB pixel / 255 inputs. This training
entry point deliberately retains that convention. New validation results use a
different, deduplicated split and must not be compared as identical experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform

from data_loader import DEFAULT_DATA_DIR, make_generators, prepare_split, save_manifest, sha256_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/training"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet",
                        help="Use ImageNet features, or random frozen features for offline smoke tests.")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be greater than 0 and less than 1.")
    return args


def build_model(num_classes: int, weights: str = "imagenet"):
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        weights=None if weights == "none" else "imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base.trainable = False
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    features = base(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.35)(features)
    features = tf.keras.layers.Dense(256, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.35)(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)
    return tf.keras.Model(inputs, outputs)


def compute_class_weight(labels, num_classes: int):
    """Count integer labels without loading or augmenting image pixels."""
    import numpy as np

    counts = np.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or np.any(counts == 0):
        raise ValueError("Every training class must have at least one image.")
    return counts, {index: float(counts.sum() / (num_classes * count)) for index, count in enumerate(counts)}


def save_plots(history: dict, confusion, labels: list[str], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        epochs = range(1, len(history[metric]) + 1)
        axis.plot(epochs, history[metric], label="training")
        axis.plot(epochs, history[f"val_{metric}"], label="validation")
        axis.set(title=title, xlabel="Epoch")
        axis.legend()
        axis.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 10))
    heatmap = axis.imshow(confusion, cmap="Blues")
    ticks = list(range(len(labels)))
    axis.set(xticks=ticks, yticks=ticks, xticklabels=labels, yticklabels=labels,
             xlabel="Predicted label", ylabel="True label", title="Validation confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(axis.get_yticklabels(), fontsize=7)
    fig.colorbar(heatmap, ax=axis)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Output directory must be empty. Choose a new --output-dir for each run.")
    split = prepare_split(args.data_dir, args.validation_fraction, args.seed)

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import numpy as np
    import tensorflow as tf
    import sklearn
    from sklearn.metrics import classification_report, confusion_matrix

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(split, output_dir / "split_manifest.json")
    class_indices = {label: index for index, label in enumerate(split.classes)}
    (output_dir / "class_indices.json").write_text(json.dumps(class_indices, indent=2) + "\n", encoding="utf-8")
    train, validation = make_generators(args.data_dir, split, args.batch_size)
    counts, class_weight = compute_class_weight(train.classes, len(split.classes))
    print("Training counts:", dict(zip(split.classes, counts.tolist())))
    print(f"Excluded {len(split.duplicates)} byte-identical duplicate images.")
    model = build_model(len(split.classes), args.weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    best_path = output_dir / "best_multiclass.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
    ]
    result = model.fit(train, validation_data=validation, epochs=args.epochs,
                       callbacks=callbacks, class_weight=class_weight, verbose=2)
    best = tf.keras.models.load_model(best_path, compile=False)
    final_path = output_dir / "model_multiclass_final.keras"
    best.save(final_path)
    validation.reset()
    probabilities = best.predict(validation, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    indices = list(range(len(split.classes)))
    report = classification_report(validation.classes, predictions, labels=indices,
                                   target_names=list(split.classes), digits=4, zero_division=0)
    report_dict = classification_report(validation.classes, predictions, labels=indices,
                                        target_names=list(split.classes), output_dict=True, zero_division=0)
    confusion = confusion_matrix(validation.classes, predictions, labels=indices)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    history = {key: [float(value) for value in values] for key, values in result.history.items()}
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    np.save(output_dir / "validation_y_true.npy", validation.classes)
    np.save(output_dir / "validation_y_pred.npy", predictions)
    save_plots(history, confusion, list(split.classes), output_dir)
    metadata = {
        "schema_version": 1,
        "seed": args.seed,
        "requested_epochs": args.epochs,
        "completed_epochs": len(history["loss"]),
        "batch_size": args.batch_size,
        "validation_fraction": args.validation_fraction,
        "initial_backbone_weights": args.weights,
        "backbone_trainable": False,
        "preprocessing": "RGB, 224x224 nearest-neighbor resize, pixel / 255.0",
        "validation_augmentation": False,
        "checkpoint_selection": "minimum validation loss",
        "split_manifest_sha256": sha256_file(output_dir / "split_manifest.json"),
        "model_sha256": sha256_file(final_path),
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": getattr(tf.keras, "__version__", "unknown"),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "validation_report": report_dict,
        "limitation": "Validation is used for model selection; this is not independent test performance.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(report)
    print(f"Saved model, split manifest, run metadata, and validation reports to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        raise SystemExit(f"Training failed: {error}") from error
