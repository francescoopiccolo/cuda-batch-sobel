#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
make
mkdir -p output
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > output/device.txt
fi
./bin/sobel data output
