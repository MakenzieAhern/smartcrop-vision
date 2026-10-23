"""Reserved entry point: a validated fine-tuning workflow is not implemented.

The original helper only changed trainable flags and saved the unchanged model;
it never called model.fit. It must not produce a checkpoint labelled fine-tuned.
A future implementation needs explicit training/validation data, must unfreeze
layers inside the nested MobileNetV2 backbone (keeping BatchNorm frozen), and
must evaluate the trained checkpoint before claiming an improvement.
"""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    parser.error(
        "Fine-tuning is not implemented; no training was run and no checkpoint "
        "was saved. Use model_training_multiclass.py for the supported training "
        "workflow, or implement and validate model.fit with explicit training "
        "and validation data before producing a fine-tuned model."
    )


if __name__ == "__main__":
    main()
