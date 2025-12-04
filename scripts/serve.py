#!/usr/bin/env python3
"""
HAVOC-2B Inference Server

Usage:
    python scripts/serve.py --config configs/inference/default_inference.yaml
    python scripts/serve.py --checkpoint checkpoints/checkpoint_step_10000 --port 8080

This script starts the FastAPI server for serving the HAVOC-2B model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn
import yaml

from havoc_core.config import HavocConfig, InferenceConfig
from havoc_inference.server import create_app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config_from_yaml(config_path: str) -> InferenceConfig:
    """Load inference configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Parse model config
    if "model" in config_dict:
        model_config_dict = config_dict["model"]
        from havoc_core.config import AttentionConfig, MLPConfig

        # Parse nested attention config
        if "attention" in model_config_dict:
            attn_dict = model_config_dict["attention"]
            attention_config = AttentionConfig(**attn_dict)
            model_config_dict["attention"] = attention_config

        # Parse nested MLP config
        if "mlp" in model_config_dict:
            mlp_dict = model_config_dict["mlp"]
            mlp_config = MLPConfig(**mlp_dict)
            model_config_dict["mlp"] = mlp_config

        model_config = HavocConfig(**model_config_dict)
        config_dict["model_config"] = model_config
        del config_dict["model"]

    return InferenceConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description="Serve HAVOC-2B model via FastAPI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference/default_inference.yaml",
        help="Path to inference configuration YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. Overrides config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host. Overrides config.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port. Overrides config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu). Overrides config.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config_from_yaml(args.config)

    # Override config with CLI arguments
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.device:
        config.device = args.device

    # Print configuration
    print("\n" + "=" * 80)
    print("HAVOC-2B Inference Server Configuration")
    print("=" * 80)
    print(f"Model: {config.model_config.num_layers} layers, {config.model_config.d_model} d_model")
    print(f"Vocab size: {config.model_config.vocab_size}")
    print(f"Checkpoint: {config.checkpoint_path or 'None (random weights)'}")
    print(f"Device: {config.device}")
    print(f"Mixed precision: {config.use_amp} ({config.amp_dtype if config.use_amp else 'N/A'})")
    print(f"Server: http://{config.host}:{config.port}")
    print(f"Max new tokens: {config.max_new_tokens}")
    print(f"Temperature: {config.temperature}")
    print(f"Top-p: {config.top_p}")
    print(f"Top-k: {config.top_k}")
    print("=" * 80 + "\n")

    # Create FastAPI app
    app = create_app(config)

    # Start server
    logger.info(f"Starting server on {config.host}:{config.port}")
    logger.info("Available endpoints:")
    logger.info(f"  - Root:       http://{config.host}:{config.port}/")
    logger.info(f"  - Health:     http://{config.host}:{config.port}/health")
    logger.info(f"  - Ready:      http://{config.host}:{config.port}/health/ready")
    logger.info(f"  - Completion: http://{config.host}:{config.port}/completion")
    logger.info(f"  - Generate:   http://{config.host}:{config.port}/generate")
    logger.info(f"  - Chat:       http://{config.host}:{config.port}/chat")
    logger.info(f"  - API Docs:   http://{config.host}:{config.port}/docs")
    logger.info("")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
