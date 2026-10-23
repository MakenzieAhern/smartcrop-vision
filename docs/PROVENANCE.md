# Project provenance

SmartCrop Vision was created by **Yuchen Deng** ([MakenzieAhern](https://github.com/MakenzieAhern)). The original implementation—including data preparation, model training, inference scripts, evaluation, and visualization—was completed by Yuchen Deng within a university group-project context.

This portfolio edition originates from `ENGG2112-group/ENGG2112-group-project` on the University of Sydney GitHub Enterprise server, at commit `a63770850f6fd16f0120296698431a0db286e4d1`. The source snapshot was retrieved and the commit verified on **23 October 2026**. The original ZIP is preserved locally. The public repository begins with a curated snapshot rather than republishing the original dataset-heavy Git history.

The portfolio maintenance pass adds documented setup, regression tests, repeatable evaluation, duplicate-aware data splitting, and repairs to inference/explanation/export tooling. Supplied trained weights remain original artifacts; no new full-dataset training result is claimed.

Third-party components include TensorFlow/Keras, the ImageNet-pretrained MobileNetV2 backbone, and PlantVillage images. Dataset and library authorship is separate from project implementation. See [data notes](DATA.md) and [the model card](MODEL_CARD.md).

The imported project did not include a license. No new blanket license is assigned to its code, weights, or data in this migration. Third-party assets remain subject to their respective source terms.
