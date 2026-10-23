"""Single-image inference with the preprocessing used by the saved model."""

import argparse
import os

import numpy as np
from PIL import Image, UnidentifiedImageError

if __package__:
    from .labels import DEFAULT_LABELS, PROJECT_ROOT, load_labels
else:
    from labels import DEFAULT_LABELS, PROJECT_ROOT, load_labels

MODEL = os.environ.get("SMARTCROP_MODEL", str(PROJECT_ROOT / "model_multiclass_final.keras"))
IMG_SIZE = (224, 224)


def preprocess(path):
    """Read RGB, resize with nearest sampling, rescale to [0, 1], and batch.

    The original ImageDataGenerator uses Pillow and its default nearest
    interpolation. Its rescale=1/255 is part of this checkpoint's contract;
    MobileNetV2's usual [-1, 1] preprocessing would change its inputs.
    """
    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.NEAREST)
            array = np.asarray(image, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Cannot read image: {path}") from exc
    array *= 1.0 / 255.0
    return array[np.newaxis, ...]


def topk(probs, labels, k=3):
    """Rank output probabilities without silently accepting a mismatched map."""
    probs = np.asarray(probs)
    if probs.ndim != 1 or len(probs) != len(labels) or not len(labels):
        raise ValueError("Prediction vector must have one value per label.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("Prediction vector contains non-finite values.")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("top-k must be a positive integer.")
    indices = np.argsort(-probs, kind="stable")[:k]
    return [(labels[index], float(probs[index])) for index in indices]


def load_inference_model(path=MODEL, labels=None):
    """Load a trusted local Keras checkpoint without restoring its optimizer."""
    from tensorflow.keras.models import load_model

    model = load_model(path, compile=False)
    if labels is not None and model.output_shape[-1] != len(labels):
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs, but label map has {len(labels)} classes."
        )
    return model


def predict_probabilities(model, images, labels):
    """Predict one batch and validate the class dimension and softmax output."""
    probabilities = np.asarray(model.predict(images, verbose=0))
    expected = (len(images), len(labels))
    if probabilities.shape != expected:
        raise ValueError(f"Model returned shape {probabilities.shape}; expected {expected}.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite probabilities.")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Model must return probabilities in [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("Model probabilities must sum to 1 for each image.")
    return probabilities


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict plant leaf disease from an image.")
    parser.add_argument("image", help="Path to a leaf image.")
    parser.add_argument("--model", default=MODEL, help="Path to a trusted Keras checkpoint.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="JSON class-to-index map.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to display.")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    labels = load_labels(args.labels)
    model = load_inference_model(args.model, labels)
    probabilities = predict_probabilities(model, preprocess(args.image), labels)[0]
    predictions = topk(probabilities, labels, args.top_k)
    print(f"\nTop-{len(predictions)} predictions:")
    for name, probability in predictions:
        print(f"  {name:<45} {probability:.4f}")


if __name__ == "__main__":
    main()
