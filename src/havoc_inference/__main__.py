"""Entry point for running inference server as a module: python -m havoc_inference

This is a convenience wrapper around scripts/serve.py for direct module execution.
For full server capabilities, use scripts/serve.py directly.
"""

import sys

print("To run inference server, please use: python scripts/serve.py --config <config_file>")
print("Example: python scripts/serve.py --config configs/inference/default_inference.yaml")
sys.exit(1)
