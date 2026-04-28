#!/bin/bash
# CUDA MPS (Multi-Process Service) Setup Script
# Enables multiple GPU processes to share the GPU efficiently without contention.
# Run as root or with sudo.

set -e

MPS_PIPE_DIR="/tmp/nvidia-mps"
MPS_LOG_DIR="/tmp/nvidia-log"

echo "=== CUDA MPS Setup ==="

# Check if MPS is already running
if echo "get_server_list" | nvidia-cuda-mps-control 2>/dev/null | grep -q ""; then
    echo "MPS daemon is already running. Stopping it first..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 1
fi

# Set GPU to EXCLUSIVE_PROCESS mode (required for MPS)
echo "Setting GPU 0 to EXCLUSIVE_PROCESS compute mode..."
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# Create pipe and log directories
mkdir -p "$MPS_PIPE_DIR" "$MPS_LOG_DIR"
chmod 777 "$MPS_PIPE_DIR" "$MPS_LOG_DIR"

# Start MPS daemon
echo "Starting MPS control daemon..."
export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR"
export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR"
nvidia-cuda-mps-control -d

sleep 1

# Verify MPS is running
if ps aux | grep -v grep | grep nvidia-cuda-mps > /dev/null; then
    echo "✓ MPS daemon is running"
    echo "  Pipe directory: $MPS_PIPE_DIR"
    echo "  Log directory:  $MPS_LOG_DIR"
else
    echo "✗ MPS daemon failed to start!"
    exit 1
fi

echo ""
echo "=== MPS Setup Complete ==="
echo "GPU compute mode: EXCLUSIVE_PROCESS"
echo "Docker containers must mount: -v /tmp/nvidia-mps:/tmp/nvidia-mps"
echo "Docker containers must use:   --ipc=host"
