"""Regression coverage for preprocessing, class mapping and honest evaluation."""

import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts.eval_folder import collect_files, evaluate_folder, save_results
from scripts.labels import DEFAULT_LABELS, PROJECT_ROOT, load_labels
from scripts.plot_confusion import plot_cm
from scripts.predict_one import predict_probabilities, preprocess, topk


def write_image(path, value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((9, 7, 3), value, dtype=np.uint8)).save(path)
    return path


def test_preprocessing_matches_training_generator(tmp_path):
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    pixels = np.random.default_rng(42).integers(0, 256, (41, 63, 3), dtype=np.uint8)
    path = tmp_path / "leaf.png"
    Image.fromarray(pixels).save(path)
    expected = img_to_array(load_img(path, target_size=(224, 224), interpolation="nearest"))
    expected *= 1.0 / 255.0
    actual = preprocess(path)
    assert actual.dtype == np.float32
    assert actual.shape == (1, 224, 224, 3)
    np.testing.assert_array_equal(actual[0], expected)


def test_preprocess_converts_grayscale_and_rejects_unreadable_images(tmp_path):
    path = tmp_path / "gray.png"
    Image.fromarray(np.full((6, 6), 128, dtype=np.uint8)).save(path)
    image = preprocess(path)
    np.testing.assert_array_equal(image[..., 0], image[..., 1])
    assert image.max() <= 1 and image.min() >= 0
    with pytest.raises(ValueError, match="Cannot read image"):
        preprocess(tmp_path / "missing.png")


@pytest.mark.parametrize("mapping", [{}, {"a": 0, "b": 0}, {"a": 1}, {"a": True}, {"a": 0.0}, ["a"]])
def test_invalid_label_map_is_rejected(tmp_path, mapping):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError):
        load_labels(path)


def test_labels_and_default_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    labels = load_labels()
    assert len(labels) == 15
    assert labels[0] == "Pepper__bell___Bacterial_spot"
    assert DEFAULT_LABELS.is_absolute()


def test_topk_preserves_label_order_and_rejects_mismatches():
    assert topk([0.1, 0.8, 0.1], ["a", "b", "c"], 8) == [("b", 0.8), ("a", 0.1), ("c", 0.1)]
    with pytest.raises(ValueError, match="one value per label"):
        topk([0.5, 0.5], ["a"])
    with pytest.raises(ValueError, match="non-finite"):
        topk([np.nan], ["a"])
    with pytest.raises(ValueError, match="positive integer"):
        topk([1.0], ["a"], 0)


class ConstantModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.batch_lengths = []

    def predict(self, images, verbose=0):
        self.batch_lengths.append(len(images))
        return np.tile(self.probabilities, (len(images), 1))


@pytest.mark.parametrize("probabilities", [[1.0], [np.nan, 0.0], [-0.1, 1.1], [0.2, 0.2]])
def test_invalid_model_predictions_fail(probabilities):
    with pytest.raises(ValueError):
        predict_probabilities(ConstantModel(probabilities), np.zeros((2, 224, 224, 3)), ["a", "b"])


def test_collect_files_is_deterministic_and_handles_mixed_case_extensions(tmp_path):
    for index in range(5):
        write_image(tmp_path / "a" / f"{index}.PNG")
    first, count = collect_files(tmp_path, ["a"], per_class=3)
    second, _ = collect_files(tmp_path, ["a"], per_class=3)
    assert first == second
    assert count == 3
    assert collect_files(tmp_path, ["a"], per_class=0)[1] == 5
    write_image(tmp_path / "unknown" / "leaf.png")
    with pytest.raises(ValueError, match="Unknown class folder"):
        collect_files(tmp_path, ["a"])


def test_evaluation_keeps_predictions_into_classes_absent_from_ground_truth(tmp_path):
    for index in range(3):
        write_image(tmp_path / "data" / "a" / f"{index}.png")
    model = ConstantModel([0.0, 1.0])
    results = evaluate_folder(model, tmp_path / "data", ["a", "b"], per_class=0, batch_size=2)
    assert model.batch_lengths == [2, 1]
    assert results["confusion_matrix"] == [[0, 3], [0, 0]]
    assert np.sum(results["confusion_matrix"]) == results["sample_count"] == 3
    assert results["accuracy"] == 0
    assert results["label_indices"] == [0, 1]
    assert results["classification_report"]["a"]["recall"] == 0
    destination = tmp_path / "results"
    save_results(results, destination)
    assert json.loads((destination / "evaluation.json").read_text())["sample_count"] == 3
    np.testing.assert_array_equal(np.load(destination / "mini_eval_y_pred.npy"), [1, 1, 1])
    matrix = plot_cm([0, 0, 0], [1, 1, 1], out=destination / "confusion.png", labels=["a", "b"])
    np.testing.assert_array_equal(matrix, [[0, 3], [0, 0]])
    assert (destination / "confusion.png").stat().st_size > 0


def test_corrupt_selected_image_fails_instead_of_changing_evaluation_total(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "bad.jpg").write_text("not an image")
    with pytest.raises(ValueError, match="Cannot read image"):
        evaluate_folder(ConstantModel([1.0]), tmp_path, ["a"], per_class=0)


@pytest.mark.parametrize("script", ["predict_one", "demo", "eval_folder", "plot_confusion_from_npy"])
def test_script_and_module_entrypoints_show_help(script, tmp_path):
    direct = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / f"{script}.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    module = subprocess.run(
        [sys.executable, "-m", f"scripts.{script}", "--help"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert direct.returncode == 0, direct.stderr
    assert module.returncode == 0, module.stderr
    assert "usage:" in direct.stdout and "usage:" in module.stdout
