# Data and reproducibility

## Image classification

The original project contains a PlantVillage subset: **20,638 images in 15 labels**, covering bell pepper, potato, and tomato. See `class_indices.json` for the exact label names and order expected by the supplied model.

The full image corpus and duplicated `mini_eval/` directory are kept outside Git. A single attributed sample in `examples/` supports a quick inference demo; its original relative path and SHA-256 are in `examples/provenance.json`. It belongs to the source dataset and is not a held-out test example.

For new experiments, obtain images from the [PlantVillage authors' repository](https://github.com/spMohanty/PlantVillage-Dataset), which also documents leaf-group-aware splits. The original project's downloaded subset/mirror was not recorded, so downloading the current upstream corpus does **not** guarantee byte-for-byte reproduction of the archived training run. Match supported classes deliberately; do not treat a newly sampled dataset as the original holdout.

Place an authorized local copy in this structure:

```text
archive/PlantVillage/
  Pepper__bell___Bacterial_spot/
  Pepper__bell___healthy/
  Potato___Early_blight/
  ...the remaining labels in class_indices.json...
  Tomato_healthy/
```

The training CLI accepts any explicit `--data-dir` with class subdirectories. New runs remove exact duplicate images before a seeded stratified split and save a manifest. Exact-byte deduplication cannot detect different photographs of the same leaf: leaf/group-based splitting and a separate field-image test set are still needed for a stronger generalization claim.

## Optional tabular exploration

`data_overview.ipynb` and `data_cleaning.ipynb` inspect and clean four local CSV files. They are exploratory notebooks, not trained crop-recommendation or crop-yield models. Their source download URLs and redistribution terms were not recorded in the imported project, so the CSVs are not republished.

| Original filename | Rows | Role |
| --- | ---: | --- |
| `Crop_recommendation.csv` | 2,200 | Crop-condition table |
| `agriculture_dataset.csv` | 50 | Agricultural observations |
| `data_season.csv` | 3,158 | Seasonal observations |
| `crop_yield.csv` | 1,000,000 | Yield observations |

The notebooks require these files at the repository root, with pandas installed. Their stored output cells have been cleared. Cleaning normalizes column names, drops exact duplicate rows, and removes entirely empty rows. These notebooks are optional; image inference and automated tests do not require the CSVs.

## Reference

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). [Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419). *Frontiers in Plant Science*, 7, 1419.
