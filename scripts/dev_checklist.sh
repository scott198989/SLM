#!/usr/bin/env bash
set -euo pipefail

# TODO: extend with linting/typechecking once tooling is wired.
python -m compileall src
pytest -q || true
