#!/bin/bash
# Stop CUDA MPS and reset GPU to default compute mode

set -e

echo "=== Stopping CUDA MPS ==="

# Stop MPS daemon
echo "Stopping MPS daemon..."
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 1

# Reset GPU to default compute mode
echo "Resetting GPU 0 to DEFAULT compute mode..."
nvidia-smi -i 0 -c DEFAULT

echo "=== MPS Stopped ==="
