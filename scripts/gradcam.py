"""Visualize class-sensitive activations in the bundled MobileNetV2 classifier."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

if __package__:
    from .predict_one import MODEL, PROJECT_ROOT, preprocess
else:
    from predict_one import MODEL, PROJECT_ROOT, preprocess

BASE_NAME = "mobilenetv2_1.00_224"
TARGET_LAYER_NAME = "out_relu"


def make_gradcam_heatmap(
    img_array, model, class_index=None, base_name=BASE_NAME,
    target_layer_name=TARGET_LAYER_NAME,
):
    """Return a normalized heatmap and class index for a single input image.

    The supported architecture is a sequential outer classifier containing a
    nested convolutional backbone, as used by the bundled checkpoint. Reusing
    its layers connects backbone activations to the classifier head; an inner
    layer's original symbolic output belongs to a different graph.
    """
    image = tf.convert_to_tensor(img_array, dtype=tf.float32)
    if image.shape.rank != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM requires one image with shape (1, H, W, C).")
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("Grad-CAM requires a single-input, single-output classifier.")
    base = model.get_layer(base_name)
    target = base.get_layer(target_layer_name)
    extractor = keras.Model(base.input, [target.output, base.output])
    inputs = keras.Input(shape=model.input_shape[1:])
    value = inputs
    activations = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if layer is base:
            activations, value = extractor(value, training=False)
        else:
            value = layer(value, training=False)
    grad_model = keras.Model(inputs, [activations, value])

    with tf.GradientTape() as tape:
        # The backbone is frozen: watching inputs records its operations too.
        tape.watch(image)
        conv_output, predictions = grad_model(image, training=False)
        index = int(tf.argmax(predictions[0])) if class_index is None else class_index
        if not isinstance(index, (int, np.integer)) or not 0 <= index < predictions.shape[-1]:
            raise ValueError(f"class_index must be in [0, {predictions.shape[-1] - 1}].")
        score = predictions[:, index]

    gradients = tape.gradient(score, conv_output)
    if gradients is None:
        raise ValueError("The selected layer is not connected to the class score.")
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))
    heatmap = tf.nn.relu(tf.reduce_sum(weights * conv_output[0], axis=-1))
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return heatmap.numpy(), int(index)


def overlay_heatmap(rgb, heatmap, alpha=0.4):
    """Blend a [0, 1] heatmap with RGB input; return an OpenCV BGR uint8 image."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")
    if np.issubdtype(rgb.dtype, np.floating):
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("Floating-point RGB pixels must be finite and in [0, 1].")
        rgb = np.round(rgb * 255).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        raise ValueError("RGB input must be normalized floating-point or uint8.")
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("Expected a finite two-dimensional heatmap.")
    heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0, 1)), cv2.COLORMAP_JET)
    return cv2.addWeighted(colored, alpha, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 1 - alpha, 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/gradcam_overlay.png")
    parser.add_argument("--class-index", type=int, help="Default: the predicted class.")
    parser.add_argument("--layer", default=TARGET_LAYER_NAME)
    args = parser.parse_args(argv)
    model = keras.models.load_model(args.model, compile=False)
    image = preprocess(args.image)
    heatmap, class_index = make_gradcam_heatmap(
        image, model, class_index=args.class_index, target_layer_name=args.layer,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), overlay_heatmap(image[0], heatmap)):
        raise OSError(f"Cannot write overlay: {args.output}")
    print(f"Saved: {args.output} (class index {class_index}, layer {args.layer})")


if __name__ == "__main__":
    main()
