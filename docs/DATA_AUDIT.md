# Dataset audit and evaluation boundaries

This audit was performed on the imported source snapshot, commit
`a63770850f6fd16f0120296698431a0db286e4d1`. The source ZIP was preserved locally.
The full datasets and exploratory CSV files are not included in the portfolio
repository. Counts below describe that local snapshot, not a fresh download.

## Reproduce the audit

The command needs only the Python standard library. Provide the same data layout:

```text
archive/PlantVillage/<class-name>/<image-file>
mini_eval/<class-name>/<image-file>
```

```bash
python -m scripts.audit_dataset \
  --data-dir archive/PlantVillage \
  --mini-dir mini_eval \
  --output runs/data-audit.json
```

Omit `--mini-dir` when the historical mini-evaluation folder is unavailable.
The JSON contains a reproducible inventory fingerprint, class counts, duplicate
paths, reconstructed original split overlap, and the revised split counts.
It contains relative image paths rather than workstation paths.

The audited inventory fingerprint is
`153dbdf840682db186db9c7e9a7a02ae6e233b51700ebc30df6a5a2537e770ac`.

## Imported image inventory

| Label | Images |
| --- | ---: |
| Pepper__bell___Bacterial_spot | 997 |
| Pepper__bell___healthy | 1,478 |
| Potato___Early_blight | 1,000 |
| Potato___Late_blight | 1,000 |
| Potato___healthy | 152 |
| Tomato_Bacterial_spot | 2,127 |
| Tomato_Early_blight | 1,000 |
| Tomato_Late_blight | 1,909 |
| Tomato_Leaf_Mold | 952 |
| Tomato_Septoria_leaf_spot | 1,771 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 1,676 |
| Tomato__Target_Spot | 1,404 |
| Tomato__Tomato_YellowLeaf__Curl_Virus | 3,208 |
| Tomato__Tomato_mosaic_virus | 373 |
| Tomato_healthy | 1,591 |
| **Total** | **20,638** |

There are **20,624 distinct SHA-256 hashes**: 14 same-label duplicate pairs.
The original Keras directory split allocated the first 15% of sorted filenames
within each class to validation: **17,549 training / 3,089 validation images**.
**Three byte-identical pairs cross those partitions.** This is a measurable
source of leakage; its contribution to the historical score is unknown.

The historical `mini_eval` folder contains **4,346 images**: 1,000 tomato early
blight, 1,756 tomato late blight, and 1,590 healthy tomato images. **All 4,346
match an image in the source dataset by file hash.** It is not an independent
holdout. The saved mini report sampled only 30 images per class, for 90 total.

Byte hashes detect exact file copies only. Related leaves, alternate captures,
resized images, recompressed copies, and shared collection sessions can still
cross an image-level split. A rigorous future benchmark needs source-level
separation and an independently collected test set.

## What the historical metrics mean

The imported full classification report records **87.80% accuracy** and
**0.8607 macro F1** across 3,089 validation images. The original code applies
random augmentation to validation as well as training and selects checkpoints
using validation performance. It does not record a complete experiment manifest.
Therefore this is a **historical validation result**, not an independent test
benchmark or a reproducible new measurement.

The imported 90-image mini report has **70 correct predictions (77.78%)**.
Its three-class confusion matrix excludes predictions assigned to other model
classes. Its class-restricted precision is not overall accuracy, and the source
overlap prevents using it as an independent test score.

Saved Keras metadata identifies version **3.11.3**. The final model was saved on
**2025-10-20** and the best checkpoint on **2025-10-22**. Both have the expected
15-output MobileNetV2 architecture. The differing dates and missing run manifest
mean that the imported report cannot be conclusively tied to one supplied model.

## Revised training protocol

`model_training_multiclass.py` now:

1. Inventories and hashes supported image files before training.
2. Keeps the lexicographically first file for each byte hash and rejects
   identical images assigned conflicting labels.
3. Uses a fixed seed and an independent shuffle within each class; every class
   must have at least two unique images to appear in both partitions.
4. Augments training images only. Validation has a stable order and no random
   transformations. Both use RGB, a 224 × 224 nearest-neighbor resize, and
   **pixel / 255.0**, preserving the supplied checkpoints' input convention.
5. Counts labels directly for class weights, records the complete split with
   file hashes, and selects the saved model by minimum validation loss.
6. Saves the model, class map, history, validation predictions, charts, and a run
   manifest linking the model hash to the split hash and runtime versions.

On this snapshot, the revised split contains **17,537 training / 3,087 validation
images**, with **zero shared byte hashes** after removing 14 duplicate files.

```bash
python model_training_multiclass.py \
  --data-dir archive/PlantVillage \
  --output-dir runs/experiment-01 \
  --epochs 12 --batch-size 32 --seed 42 --weights imagenet
```

The output directory must be empty to prevent mixing experiments. ImageNet
weights download on first use unless already cached. `--weights none` uses a
random frozen backbone for an offline smoke test; it is not a useful pretrained
classifier. `model_training.py` is a compatibility alias to the revised pipeline.
The original `model_binary.h5` naming was misleading: its training code was
multiclass, not a binary healthy/stressed classifier.

These changes do not retroactively alter the supplied model or its history.
They define a new experiment; revised split scores must be labelled separately.
Seeds and deterministic TensorFlow operations improve repeatability, but do not
guarantee identical floating-point results across hardware/library versions.
Validation remains part of model selection and is not a final independent test.

The revised pipeline was also exercised with eight seeded synthetic RGB images
across two classes, using `--weights none --epochs 1 --batch-size 2`. It completed
training, saved/reloaded its checkpoint, and produced the expected reports and
hash-linked manifests. This checks execution only; synthetic-image accuracy is
not a model-quality result. See [training smoke evidence](../reports/verification/training_smoke.json)
and the [machine-readable data audit](../reports/verification/data_audit.json).

## Exploratory tables

The original notebooks inspect and clean four separate agricultural tables:

| File | Rows | Role |
| --- | ---: | --- |
| Crop_recommendation.csv | 2,200 | Nutrient/weather values and crop labels |
| agriculture_dataset.csv | 50 | Farm inputs, water use, and yield |
| data_season.csv | 3,158 | Seasonal environmental and crop records |
| crop_yield.csv | 1,000,000 | Regional inputs and crop-yield records |

Cleaning standardizes column names, removes exact duplicate rows, and removes
fully empty rows. The saved cleaned versions retain the row counts above.
These tables are not inputs to the image classifier. No trained crop-selection
or yield-regression model is present, and those capabilities are not claimed.
The imported snapshot did not include dataset source URLs or license files;
obtain data from its original provider and check the provider's current terms.
