"""Entry point for running training as a module: python -m havoc_training

This is a convenience wrapper around scripts/train.py for direct module execution.
For full training capabilities, use scripts/train.py directly.
"""

import sys

print("To run training, please use: python scripts/train.py --config <config_file>")
print("Example: python scripts/train.py --config configs/training/default_training.yaml")
sys.exit(1)
