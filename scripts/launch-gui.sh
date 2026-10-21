#!/usr/bin/env bash
# OpenMemGPU Desktop Launcher Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Launching OpenMemGPU Unified Virtual VRAM Dashboard ==="

# Prefer Wayland if available, otherwise X11
if [ -n "$WAYLAND_DISPLAY" ]; then
    export WINIT_UNIX_BACKEND=wayland
    echo "Display Server: Wayland ($WAYLAND_DISPLAY)"
elif [ -n "$DISPLAY" ]; then
    export WINIT_UNIX_BACKEND=x11
    echo "Display Server: X11 ($DISPLAY)"
fi

# Ensure release binary exists; if not, build it
RELEASE_BIN="${PROJECT_ROOT}/target/release/openmemgpu-gui"
if [ ! -f "$RELEASE_BIN" ]; then
    echo "Building release binary..."
    cargo build --release -p openmemgpu-gui --manifest-path "${PROJECT_ROOT}/Cargo.toml"
fi

echo "Starting OpenMemGPU GUI..."
exec "$RELEASE_BIN" "$@"
