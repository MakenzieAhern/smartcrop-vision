"""Legacy command alias for the supported multiclass training pipeline.

The original script wrote ``model_binary.h5`` but actually trained one output
per dataset class. That artifact is a legacy multiclass baseline, not a binary
healthy/stressed classifier. New runs use model_training_multiclass.py.
"""

from model_training_multiclass import main


if __name__ == "__main__":
    print("Legacy entry point: using the reproducible multiclass training pipeline.")
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        raise SystemExit(f"Training failed: {error}") from error
