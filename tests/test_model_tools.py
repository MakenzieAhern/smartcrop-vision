"""Regression coverage for explanations, deployment, and the unfinished helper."""

from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import tensorflow as tf

from scripts.export_tflite import export_model
from scripts.gradcam import make_gradcam_heatmap, overlay_heatmap


@pytest.fixture
def nested_classifier():
    inputs = tf.keras.Input((4, 4, 3))
    features = tf.keras.layers.Conv2D(
        1, 1, use_bias=False, kernel_initializer="ones", name="out_relu",
    )(inputs)
    backbone = tf.keras.Model(inputs, features, name="mobilenetv2_1.00_224")
    backbone.trainable = False
    outer_input = tf.keras.Input((4, 4, 3))
    pooled = tf.keras.layers.GlobalAveragePooling2D()(backbone(outer_input))
    output = tf.keras.layers.Dense(2, activation="softmax", use_bias=False)(pooled)
    model = tf.keras.Model(outer_input, output)
    model.layers[-1].set_weights([np.array([[1, -1]], dtype=np.float32)])
    return model


def test_gradcam_connects_frozen_nested_backbone(nested_classifier):
    image = np.arange(48, dtype=np.float32).reshape(1, 4, 4, 3) / 48
    heatmap, class_index = make_gradcam_heatmap(image, nested_classifier)
    expected = image[0].sum(axis=-1)
    expected /= expected.max()
    assert class_index == int(np.argmax(nested_classifier(image).numpy()[0])) == 0
    np.testing.assert_allclose(heatmap, expected, atol=1e-6)


def test_gradcam_zero_activation_has_no_nan(nested_classifier):
    heatmap, _ = make_gradcam_heatmap(np.zeros((1, 4, 4, 3)), nested_classifier)
    assert np.isfinite(heatmap).all()
    assert not heatmap.any()


def test_gradcam_rejects_invalid_class(nested_classifier):
    with pytest.raises(ValueError, match="class_index"):
        make_gradcam_heatmap(np.zeros((1, 4, 4, 3)), nested_classifier, class_index=2)


def test_overlay_supports_preprocessed_float_image():
    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    rgb[:, :, 0] = 1
    output = overlay_heatmap(rgb, np.zeros((2, 2)), alpha=0)
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output, cv2.cvtColor(np.uint8(rgb * 255), cv2.COLOR_RGB2BGR))


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_overlay_rejects_invalid_blend(alpha):
    with pytest.raises(ValueError, match="alpha"):
        overlay_heatmap(np.zeros((4, 4, 3)), np.zeros((2, 2)), alpha=alpha)


@pytest.mark.parametrize("optimize", [False, True])
def test_export_runs_in_tflite_with_prediction_parity(tmp_path, nested_classifier, optimize):
    output = export_model(nested_classifier, tmp_path / "model.tflite", optimize=optimize)
    interpreter = tf.lite.Interpreter(model_path=str(output), num_threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()[0]
    image = np.arange(48, dtype=np.float32).reshape(1, 4, 4, 3) / 48
    assert inputs["dtype"] == np.float32
    assert outputs["dtype"] == np.float32
    interpreter.set_tensor(inputs["index"], image)
    interpreter.invoke()
    np.testing.assert_allclose(
        interpreter.get_tensor(outputs["index"]), nested_classifier(image).numpy(), atol=1e-6,
    )


def test_fine_tuning_placeholder_fails_without_writing_checkpoint(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/fine_tune.py"
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "no checkpoint was saved" in result.stderr
    assert list(tmp_path.iterdir()) == []
