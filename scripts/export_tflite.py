"""Export a Keras classifier to TensorFlow Lite with float32 input/output."""

import argparse
from pathlib import Path
import tempfile

import tensorflow as tf

if __package__:
    from .predict_one import MODEL, PROJECT_ROOT
else:
    from predict_one import MODEL, PROJECT_ROOT


def export_model(model, output_path, optimize=False):
    """Export through SavedModel for compatibility with TensorFlow 2.17/Keras 3.

    Float32 weights are the default: dynamic-range quantization changed a
    predicted class in the bundled-model smoke check. Optional optimization
    quantizes weights and requires a separate accuracy evaluation. Inputs always
    require the same float32 RGB [0, 1] preprocessing as the Keras checkpoint.
    """
    output_path = Path(output_path)
    with tempfile.TemporaryDirectory(prefix="smartcrop-export-") as directory:
        model.export(directory, format="tf_saved_model", verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(directory)
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converted = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converted)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/smartcrop.tflite")
    parser.add_argument(
        "--quantize", action="store_true",
        help="Experimental dynamic-range weight quantization; validate prediction drift before use.",
    )
    args = parser.parse_args(argv)
    model = tf.keras.models.load_model(args.model, compile=False)
    output = export_model(model, args.output, optimize=args.quantize)
    print(f"Saved: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
