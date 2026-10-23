"""Regression tests for split integrity; no TensorFlow or private data required."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from data_loader import ImageRecord, inventory_images, prepare_split, save_manifest, split_records
from scripts.audit_dataset import audit_dataset


ROOT = Path(__file__).resolve().parents[1]


class DataSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "images"
        for label in ("healthy", "disease"):
            (self.root / label).mkdir(parents=True)
            for index in range(10):
                # The inventory audits bytes; pixel decoding belongs to training.
                (self.root / label / f"{index:02d}.JPG").write_bytes(f"{label}:{index}".encode())

    def test_deduplication_prevents_cross_partition_leakage(self):
        (self.root / "healthy" / "duplicate.jpg").write_bytes((self.root / "healthy" / "00.JPG").read_bytes())
        split = prepare_split(self.root, validation_fraction=0.2, seed=42)
        self.assertEqual((len(split.train), len(split.validation), len(split.duplicates)), (16, 4, 1))
        train_hashes = {record.sha256 for record in split.train}
        validation_hashes = {record.sha256 for record in split.validation}
        self.assertFalse(train_hashes & validation_hashes)
        self.assertEqual(len(train_hashes | validation_hashes), 20)
        self.assertEqual({record.label for record in split.validation}, {"healthy", "disease"})

    def test_seed_and_inventory_order_produce_stable_manifests(self):
        records = inventory_images(self.root)
        first = split_records(records, 0.2, 42)
        second = split_records(list(reversed(records)), 0.2, 42)
        self.assertEqual(first.manifest(), second.manifest())
        self.assertNotEqual(first.validation, split_records(records, 0.2, 43).validation)
        manifest_path = Path(self.temporary.name) / "output" / "manifest.json"
        save_manifest(first, manifest_path)
        saved = json.loads(manifest_path.read_text())
        self.assertEqual(saved, first.manifest())
        self.assertNotIn(str(self.root), manifest_path.read_text())

    def test_identical_bytes_with_conflicting_labels_fail(self):
        (self.root / "healthy" / "wrong_label.jpg").write_bytes((self.root / "disease" / "00.JPG").read_bytes())
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            prepare_split(self.root)
        report = audit_dataset(self.root)
        self.assertEqual(report["conflicting_label_groups"], 1)
        self.assertIn("error", report["revised_split"])

    def test_too_few_unique_images_fail(self):
        records = [ImageRecord("a/1.jpg", "a", "same"), ImageRecord("a/2.jpg", "a", "same")]
        with self.assertRaisesRegex(ValueError, "at least two unique"):
            split_records(records)

    def test_invalid_split_fraction_and_empty_dataset_fail(self):
        for fraction in (-1, 0, 1, 2):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                prepare_split(self.root, fraction)
        with self.assertRaisesRegex(ValueError, "No images"):
            split_records([])

    def test_audit_detects_original_split_leakage_and_mini_overlap(self):
        (self.root / "disease" / "99.jpg").write_bytes((self.root / "disease" / "00.JPG").read_bytes())
        mini_dir = Path(self.temporary.name) / "mini"
        (mini_dir / "disease").mkdir(parents=True)
        (mini_dir / "disease" / "copy.png").write_bytes((self.root / "disease" / "00.JPG").read_bytes())
        report = audit_dataset(self.root, mini_dir)
        self.assertEqual(report["images"], 21)
        self.assertEqual(report["unique_image_hashes"], 20)
        self.assertEqual(report["historical_filename_split"]["duplicate_groups_crossing_split"], 1)
        self.assertEqual(report["revised_split"]["overlapping_hashes"], 0)
        self.assertEqual(report["mini_eval"]["matches_source"], 1)
        self.assertTrue(report["mini_eval"]["has_exact_source_overlap"])
        self.assertFalse(report["mini_eval"]["independent_test_set"])
        (mini_dir / "disease" / "copy.png").write_bytes(b"no exact match")
        unmatched = audit_dataset(self.root, mini_dir)
        self.assertFalse(unmatched["mini_eval"]["has_exact_source_overlap"])
        self.assertIsNone(unmatched["mini_eval"]["independent_test_set"])

    def test_inventory_ignores_hidden_and_nonimage_files(self):
        (self.root / "healthy" / ".hidden.jpg").write_bytes(b"hidden")
        (self.root / "healthy" / "note.txt").write_text("not an image")
        (self.root / ".cache").mkdir()
        (self.root / ".cache" / "cache.jpg").write_bytes(b"hidden")
        self.assertEqual(len(inventory_images(self.root)), 20)

    def test_imports_do_not_train_scan_write_or_import_tensorflow(self):
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import data_loader, model_training_multiclass, model_training; "
            "assert 'tensorflow' not in sys.modules"
        )
        empty = Path(self.temporary.name) / "empty"
        empty.mkdir()
        process = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT)], cwd=empty,
                                 text=True, capture_output=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")
        self.assertEqual(list(empty.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
