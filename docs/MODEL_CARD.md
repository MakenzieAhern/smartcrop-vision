# Model card: SmartCrop Vision

## Intended use

A portfolio and research prototype for classifying isolated plant-leaf photographs into **15 known labels** for bell pepper, potato, and tomato. It demonstrates transfer learning, reproducible evaluation, and model interpretation. It does not identify arbitrary plants, detect multiple diseases, estimate disease severity, or establish a clinical/agronomic diagnosis.

## Architecture and inputs

- Input: one RGB image resized to **224 × 224** with nearest-neighbor interpolation; float32 pixels divided by 255.
- Backbone: ImageNet-pretrained **MobileNetV2**, frozen during the original first-stage training.
- Head: global average pooling → dropout 0.35 → dense 256/ReLU → dropout 0.35 → dense 15/softmax.
- Total parameters: **2,589,775**.
- Original optimization: Adam, learning rate `1e-4`, categorical cross entropy, inverse-frequency class weights, early stopping, and learning-rate reduction.
- Output order: `class_indices.json`. Softmax values are model scores, not calibrated probabilities of disease.

The supplied checkpoints were trained using `[0,1]` scaling. MobileNetV2's conventional `[-1,1]` preprocessing must **not** be substituted when using these weights. Inference now uses nearest-neighbor resizing to match the original Keras directory loader; the original standalone OpenCV scripts used bilinear resizing. This maintenance change can alter predictions on non-224-pixel inputs.

## Supplied artifacts

| Artifact | Interpretation |
| --- | --- |
| `model_multiclass_final.keras` | Original final model; default for inference and current verification |
| `checkpoints/best_multiclass.keras` | Original validation-selected checkpoint; distinct from the final model |
| `model_binary.h5` | Historical baseline filename; the original training code is multiclass |
| `reports/classification_report.txt` | Historical validation report; not a new run |
| `reports/mini_eval_report.txt` | Historical exploratory report on overlapping source images |

The final model and best-checkpoint archive metadata have different save dates. The historical reports cannot be conclusively paired with a particular run from the available records. Current verification records model SHA-256 values explicitly.

## Evaluation limitations

The original dataset has 20,638 images; the historical filename-based split used 17,549 training images and 3,089 validation images. Validation images were randomly augmented, and the same partition guided checkpoint selection. Exact duplicates also crossed that split. The historical 87.80% validation accuracy is therefore not an independent benchmark.

All 4,346 images under the original `mini_eval/` directory also occur in the main dataset. Current sampled evaluations using these images or the main source directory test that the pipeline runs and characterize checkpoint behavior; they do not measure unseen-data generalization. See [the data audit](DATA_AUDIT.md) and [verification record](VERIFICATION.md).

New training uses deterministic, byte-deduplicated partitions and unaugmented validation. Those improvements do not retroactively change the provenance of the supplied weights. Similar photos of the same leaf may remain after byte deduplication; group/leaf-aware splitting is preferable for new benchmark work.

## Interpretation and deployment

Grad-CAM shows regions influencing the selected output, not proof of a correct diagnosis. Rule-based follow-up notes are static illustrative text, not a learned treatment system. Real field photos can differ substantially in background, illumination, leaf pose, and mixed symptoms from curated images.

TensorFlow Lite export is a deployment experiment. Conversion and a numerical parity smoke test do not establish accuracy on mobile devices, latency targets, robustness, or production readiness.

## Next validation work

1. Collect or designate a genuinely independent field-image test set.
2. Split by leaf/plant/source group before augmentation and training.
3. Compare against a simple baseline and report per-class uncertainty.
4. Evaluate calibration, out-of-distribution inputs, and mobile-device latency.
