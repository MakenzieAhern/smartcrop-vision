"""Deterministic image inventory, deduplication, and training/validation loading.

Importing this module does not scan a dataset or import TensorFlow. Existing
checkpoints expect RGB images resized to 224 x 224 and scaled to [0, 1].
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff"})
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "archive" / "PlantVillage"


@dataclass(frozen=True)
class ImageRecord:
    """An image path relative to its dataset root, with its byte-level identity."""

    path: str
    label: str
    sha256: str


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[ImageRecord, ...]
    validation: tuple[ImageRecord, ...]
    duplicates: tuple[ImageRecord, ...]
    classes: tuple[str, ...]
    seed: int
    validation_fraction: float

    def manifest(self) -> dict:
        """Return a portable manifest without absolute paths or machine details."""
        return {
            "schema_version": 1,
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "split_method": "sha256 deduplication, then seeded shuffle within each class",
            "duplicate_policy": "Keep the lexicographically first path per SHA-256; reject conflicting labels",
            "preprocessing": {"color": "RGB", "image_size": [224, 224], "scale": "pixel / 255.0"},
            "class_indices": {label: index for index, label in enumerate(self.classes)},
            "counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "duplicates_excluded": len(self.duplicates),
                "train_per_class": dict(sorted(Counter(r.label for r in self.train).items())),
                "validation_per_class": dict(sorted(Counter(r.label for r in self.validation).items())),
            },
            "train": [asdict(record) for record in self.train],
            "validation": [asdict(record) for record in self.validation],
            "duplicates_excluded": [asdict(record) for record in self.duplicates],
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_images(data_dir: str | Path) -> tuple[ImageRecord, ...]:
    """List supported images in class subdirectories in stable path order.

    This verifies byte identity, not whether an image can be decoded. Hidden
    directories/files are excluded. Empty class directories are rejected.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not class_dirs:
        raise ValueError("Dataset must contain one subdirectory per class.")
    records = []
    for class_dir in class_dirs:
        images = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and not any(part.startswith(".") for part in p.relative_to(root).parts)
        )
        if not images:
            raise ValueError(f"Class has no supported images: {class_dir.name}")
        records.extend(
            ImageRecord(p.relative_to(root).as_posix(), class_dir.name, sha256_file(p))
            for p in images
        )
    return tuple(records)


def split_records(
    records: tuple[ImageRecord, ...] | list[ImageRecord],
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Deduplicate globally and create a reproducible stratified image split.

    At least two unique images per class are required so that every class is
    represented in both partitions. Image hashes cannot cross partitions.
    Related leaves, plants, or acquisition sessions are not inferred from paths.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be greater than 0 and less than 1.")
    if not records:
        raise ValueError("No images to split.")
    seen: dict[str, ImageRecord] = {}
    duplicates = []
    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda r: r.path):
        previous = seen.get(record.sha256)
        if previous is not None:
            if previous.label != record.label:
                raise ValueError(
                    f"Identical image has conflicting labels: {previous.path} and {record.path}"
                )
            duplicates.append(record)
            continue
        seen[record.sha256] = record
        by_class[record.label].append(record)

    train, validation = [], []
    for label, images in sorted(by_class.items()):
        if len(images) < 2:
            raise ValueError(f"Class needs at least two unique images: {label}")
        random.Random(f"{seed}:{label}").shuffle(images)
        n_validation = max(1, min(len(images) - 1, int(len(images) * validation_fraction)))
        validation.extend(images[:n_validation])
        train.extend(images[n_validation:])
    return DatasetSplit(
        train=tuple(sorted(train, key=lambda r: r.path)),
        validation=tuple(sorted(validation, key=lambda r: r.path)),
        duplicates=tuple(duplicates),
        classes=tuple(sorted(by_class)),
        seed=seed,
        validation_fraction=validation_fraction,
    )


def prepare_split(data_dir: str | Path, validation_fraction: float = 0.15, seed: int = 42) -> DatasetSplit:
    return split_records(inventory_images(data_dir), validation_fraction, seed)


def save_manifest(split: DatasetSplit, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(split.manifest(), indent=2) + "\n", encoding="utf-8")


def make_generators(data_dir: str | Path, split: DatasetSplit, batch_size: int = 32):
    """Create augmented training and unaugmented, ordered validation generators.

    TensorFlow and pandas are imported only when training is requested. Keep
    pixel / 255 normalization for compatibility with the supplied checkpoints.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    import pandas as pd
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    root = Path(data_dir).expanduser().resolve()
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.12,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    def frame(records):
        return pd.DataFrame({"filename": [r.path for r in records], "class": [r.label for r in records]})

    common = {
        "directory": str(root),
        "target_size": (224, 224),
        "color_mode": "rgb",
        "class_mode": "categorical",
        "classes": list(split.classes),
        "batch_size": batch_size,
        "seed": split.seed,
        "interpolation": "nearest",
    }
    train = train_datagen.flow_from_dataframe(frame(split.train), shuffle=True, **common)
    validation = validation_datagen.flow_from_dataframe(frame(split.validation), shuffle=False, **common)
    if train.samples != len(split.train) or validation.samples != len(split.validation):
        raise ValueError("Generator skipped files from the recorded split; verify image paths and formats.")
    return train, validation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the reproducible training/validation split.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    try:
        split = prepare_split(args.data_dir, args.validation_fraction, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(split.manifest()["counts"], indent=2))
