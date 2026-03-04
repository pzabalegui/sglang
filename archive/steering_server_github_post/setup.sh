#!/bin/bash
# Setup script for Steering Server
# Run as: bash setup.sh

set -e

echo "=============================================="
echo "  STEERING SERVER - SETUP"
echo "=============================================="

# Install jq if not present
echo ""
echo "[1/6] Installing jq..."
if ! command -v jq &> /dev/null; then
    if command -v apt-get &> /dev/null; then
        apt-get update -qq && apt-get install -y jq -qq
    elif command -v yum &> /dev/null; then
        yum install -y jq -q
    elif command -v brew &> /dev/null; then
        brew install jq
    else
        echo "WARNING: Could not install jq automatically. Install manually."
    fi
fi
echo "jq $(jq --version 2>/dev/null || echo 'not installed')"

# Check NVIDIA GPU
echo ""
echo "[2/6] Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "GPU OK"

# Check Python
echo ""
echo "[3/6] Checking Python..."
python3 --version
if ! python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "WARNING: Python 3.10+ recommended"
fi

# Create virtual environment
echo ""
echo "[4/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo "Virtual environment activated"

# Install dependencies
echo ""
echo "[5/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify CUDA
echo ""
echo "[6/6] Verifying CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=============================================="
echo "  SETUP COMPLETE"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Extract refusal direction (GLM-4.7-Flash, layer 24):"
echo "   python extract_and_serve.py --model zai-org/GLM-4.7-Flash --layer 24 --save-vector refusal_direction_glm47flash.pt"
echo ""
echo "2. Start server:"
echo "   python runtime_steering.py --serve --model zai-org/GLM-4.7-Flash --vector refusal_direction_glm47flash.pt --port 8000"
echo ""
echo "   With 4-bit quantization (24GB VRAM):"
echo "   python runtime_steering.py --serve --model zai-org/GLM-4.7-Flash --vector refusal_direction_glm47flash.pt --port 8000 --4bit"
echo ""
echo "3. Test:"
echo '   curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '"'"'{"prompt": "How to hack?", "steering": true}'"'"
echo ""
