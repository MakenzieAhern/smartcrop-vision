"""Audit image counts, exact duplicates, historical split leakage, and overlap.

Uses only the Python standard library. Byte hashes do not detect related leaves,
near duplicates, or differences caused by recompression/cropping.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import DEFAULT_DATA_DIR, inventory_images, split_records


def audit_dataset(data_dir: str | Path, mini_dir: str | Path | None = None,
                  validation_fraction: float = 0.15, seed: int = 42) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be greater than 0 and less than 1.")
    records = inventory_images(data_dir)
    by_hash = defaultdict(list)
    by_class = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
        by_class[record.label].append(record)

    # Reconstruct the original ImageDataGenerator per-class filename split,
    # before deduplication. This is not the revised training split.
    historical_partition = {}
    for label_records in by_class.values():
        ordered = sorted(label_records, key=lambda record: record.path)
        n_validation = int(len(ordered) * validation_fraction)
        for index, record in enumerate(ordered):
            historical_partition[record.path] = "validation" if index < n_validation else "train"
    duplicate_groups = []
    for digest, group in sorted(by_hash.items()):
        if len(group) > 1:
            duplicate_groups.append({
                "sha256": digest,
                "paths": [record.path for record in group],
                "labels": sorted({record.label for record in group}),
                "crosses_historical_split": len({historical_partition[r.path] for r in group}) > 1,
            })
    inventory_description = [{"path": r.path, "label": r.label, "sha256": r.sha256} for r in records]
    inventory_digest = hashlib.sha256(
        json.dumps(inventory_description, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": 1,
        "method": "SHA-256 of file bytes; case-insensitive image extensions; sorted relative paths",
        "inventory_sha256": inventory_digest,
        "images": len(records),
        "classes": len(by_class),
        "counts_per_class": dict(sorted(Counter(r.label for r in records).items())),
        "unique_image_hashes": len(by_hash),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_files_beyond_first": len(records) - len(by_hash),
        "conflicting_label_groups": sum(len(group["labels"]) > 1 for group in duplicate_groups),
        "historical_filename_split": {
            "validation_fraction": validation_fraction,
            "counts": dict(sorted(Counter(historical_partition.values()).items())),
            "duplicate_groups_crossing_split": sum(group["crosses_historical_split"] for group in duplicate_groups),
        },
        "duplicates": duplicate_groups,
    }
    try:
        revised_split = split_records(records, validation_fraction, seed)
        report["revised_split"] = {
            "seed": seed,
            "counts": revised_split.manifest()["counts"],
            "overlapping_hashes": len({r.sha256 for r in revised_split.train} &
                                      {r.sha256 for r in revised_split.validation}),
        }
    except ValueError as error:
        report["revised_split"] = {"error": str(error)}

    if mini_dir is not None:
        mini_records = inventory_images(mini_dir)
        overlap_by_class = {}
        for label in sorted({record.label for record in mini_records}):
            matching = [record for record in mini_records if record.label == label]
            overlap_by_class[label] = {
                "images": len(matching),
                "matches_source": sum(record.sha256 in by_hash for record in matching),
            }
        report["mini_eval"] = {
            "images": len(mini_records),
            "matches_source": sum(record.sha256 in by_hash for record in mini_records),
            "has_exact_source_overlap": any(record.sha256 in by_hash for record in mini_records),
            "unique_hashes": len({record.sha256 for record in mini_records}),
            "counts_and_overlap_per_class": overlap_by_class,
            "independent_test_set": False if any(record.sha256 in by_hash for record in mini_records) else None,
            "interpretation": (
                "Any source overlap invalidates a fully independent holdout claim; "
                "absence of exact-byte overlap alone does not establish independence."
            ),
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--mini-dir", type=Path, help="Optional folder to check against the main dataset.")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this path.")
    args = parser.parse_args(argv)
    try:
        report = audit_dataset(args.data_dir, args.mini_dir, args.validation_fraction, args.seed)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
