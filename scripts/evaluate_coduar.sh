#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/evaluate_coduar.example.yaml}"

echo "TODO: connect this wrapper to the cleaned CoDuAR evaluation entry point."
echo "Config: ${CONFIG_PATH}"
echo
echo "Expected future command:"
echo "python -m coduar.cli evaluate --config ${CONFIG_PATH}"
