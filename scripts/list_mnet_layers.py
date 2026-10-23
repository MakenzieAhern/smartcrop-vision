"""Inspect the MobileNetV2 backbone in a local checkpoint."""
import argparse

if __package__:
    from .predict_one import MODEL
else:
    from predict_one import MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL))
    args = parser.parse_args()
    from tensorflow.keras.models import load_model
    model = load_model(args.model, compile=False)
    base = next(layer for layer in model.layers if layer.name.startswith("mobilenetv2"))
    for index, layer in enumerate(base.layers):
        print(f"{index:3d}  {layer.name:40s}  trainable={layer.trainable}")


if __name__ == "__main__":
    main()
