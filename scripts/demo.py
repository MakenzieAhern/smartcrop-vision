"""Leaf classification demo with the project's static management suggestions."""

import argparse

if __package__:
    from .labels import DEFAULT_LABELS, load_labels
    from .recommend import recommend_for
    from .predict_one import MODEL, load_inference_model, predict_probabilities, preprocess, topk
else:
    from labels import DEFAULT_LABELS, load_labels
    from recommend import recommend_for
    from predict_one import MODEL, load_inference_model, predict_probabilities, preprocess, topk


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a leaf image.")
    parser.add_argument("--model", default=MODEL, help="Path to a trusted Keras checkpoint.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="JSON class-to-index map.")
    args = parser.parse_args(argv)
    labels = load_labels(args.labels)
    model = load_inference_model(args.model, labels)
    probabilities = predict_probabilities(model, preprocess(args.image), labels)[0]
    predictions = topk(probabilities, labels)
    print(f"\nTop-{len(predictions)}:")
    for name, probability in predictions:
        print(f"  {name:<45} {probability:.4f}")
    print("\nGeneral management suggestion:", recommend_for(predictions[0][0]))


if __name__ == "__main__":
    main()
