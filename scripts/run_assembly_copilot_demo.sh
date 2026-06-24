#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/demo_assembly_copilot.example.yaml}"

echo "TODO: connect this wrapper to the cleaned Assembly Copilot demo entry point."
echo "Config: ${CONFIG_PATH}"
echo
echo "Expected future command:"
echo "python -m assembly_copilot.cli demo --config ${CONFIG_PATH}"
