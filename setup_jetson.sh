#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_jetson.sh
# ───────────────
# One-shot setup script for Fish Detection App on NVIDIA Jetson Orin Nano.
# Tested on JetPack 5.1.x (Ubuntu 20.04, CUDA 11.4)
#
# Run with:
#   chmod +x setup_jetson.sh
#   ./setup_jetson.sh
#
# After setup, activate the venv:
#   source ~/fish_venv/bin/activate
#   python main.py
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on first error

VENV_DIR="$HOME/fish_venv"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Detect JetPack version ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  🐟  Fish Detection — Jetson Setup"
echo "============================================================"
echo ""

JP_VERSION=""
if [ -f /etc/nv_tegra_release ]; then
    JP_VERSION=$(cat /etc/nv_tegra_release | head -1)
    echo "  JetPack: $JP_VERSION"
fi

CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
echo "  CUDA   : ${CUDA_VERSION:-not found}"
echo "  Python : $(python3 --version)"
echo "  App dir: $APP_DIR"
echo ""

# ─── Step 1: System dependencies ─────────────────────────────────────────────
echo "━━━ Step 1: System packages ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sudo apt-get update -q
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-pyqt5 \
    pyqt5-dev-tools \
    python3-opencv \
    libopencv-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    wget
echo "  ✅ System packages installed"
echo ""

# ─── Step 2: Create virtual environment with system packages ──────────────────
echo "━━━ Step 2: Virtual environment ━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" --system-site-packages
    echo "  ✅ Created venv at $VENV_DIR"
else
    echo "  ✅ Venv already exists at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools --quiet
echo ""

# ─── Step 3: PyTorch (NVIDIA aarch64 wheel) ───────────────────────────────────
echo "━━━ Step 3: PyTorch for Jetson ━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if torch is already installed with CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    echo "  ✅ PyTorch $TORCH_VER already installed with CUDA — skipping"
else
    echo "  Installing PyTorch from NVIDIA Jetson wheel..."
    echo ""
    echo "  ⚠️  Please download the correct wheel for your JetPack version from:"
    echo "      https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo ""
    echo "  JetPack 5.1.2 (CUDA 11.4) → torch 2.1.0:"
    echo "    wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
    echo "    pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
    echo ""
    echo "  JetPack 6.x (CUDA 12.x) → torch 2.3+:"
    echo "    wget <url from NVIDIA forums for your JP6 version>"
    echo "    pip install <downloaded_wheel.whl>"
    echo ""
    echo "  After installing PyTorch, re-run this script to continue setup."
    echo ""
    read -p "  Have you already installed PyTorch? Press Enter to continue or Ctrl+C to abort: "
fi
echo ""

# ─── Step 4: torchvision ─────────────────────────────────────────────────────
echo "━━━ Step 4: torchvision ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if python3 -c "import torchvision" 2>/dev/null; then
    TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__)")
    echo "  ✅ torchvision $TV_VER already installed — skipping"
else
    echo "  Building torchvision from source (matches installed torch)..."
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
    # Map torch version to torchvision branch
    case "$TORCH_VER" in
        2.0*) TV_BRANCH="v0.15.2" ;;
        2.1*) TV_BRANCH="v0.16.2" ;;
        2.2*) TV_BRANCH="v0.17.2" ;;
        2.3*) TV_BRANCH="v0.18.1" ;;
        *)    TV_BRANCH="main" ;;
    esac
    echo "  torch $TORCH_VER → torchvision $TV_BRANCH"
    cd /tmp
    if [ ! -d "vision" ]; then
        git clone --branch "$TV_BRANCH" --depth 1 https://github.com/pytorch/vision.git
    fi
    cd vision
    pip install -e . --no-build-isolation -q
    cd "$APP_DIR"
    echo "  ✅ torchvision built and installed"
fi
echo ""

# ─── Step 5: Python packages ──────────────────────────────────────────────────
echo "━━━ Step 5: Python packages ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install -r "$APP_DIR/requirements_jetson.txt" --quiet
echo "  ✅ Python packages installed"
echo ""

# ─── Step 6: Verify CUDA ─────────────────────────────────────────────────────
echo "━━━ Step 6: Verify CUDA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import torch
print(f'  torch     : {torch.__version__}')
print(f'  CUDA      : {torch.version.cuda}')
print(f'  GPU avail : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name  : {torch.cuda.get_device_name(0)}')
"

python3 -c "
try:
    import tensorrt as trt
    print(f'  TensorRT  : {trt.__version__}')
except ImportError:
    print('  TensorRT  : not found as Python package (may still work via ultralytics)')
"
echo ""

# ─── Done ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  ✅  Setup complete!"
echo "============================================================"
echo ""
echo "  Activate venv:   source $VENV_DIR/bin/activate"
echo ""
echo "  Export model to TensorRT (do this once, on the Jetson):"
echo "    python export_model.py --format engine --half"
echo ""
echo "  Set model path and run:"
echo "    export FISH_MODEL_PATH=\$(ls ../export/best.engine 2>/dev/null || echo '../export/best.pt')"
echo "    python main.py"
echo ""
echo "  For max performance, set power mode first:"
echo "    sudo nvpmodel -m 0"
echo "    sudo jetson_clocks"
echo ""
