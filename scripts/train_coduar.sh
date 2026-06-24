#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train_coduar.example.yaml}"

echo "TODO: connect this wrapper to the cleaned CoDuAR training entry point."
echo "Config: ${CONFIG_PATH}"
echo
echo "Expected future command:"
echo "python -m coduar.cli train --config ${CONFIG_PATH}"
