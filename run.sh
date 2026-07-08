#!/usr/bin/env bash
# =============================================================================
# run.sh — Fish Detection & Counting App Launcher
# =============================================================================
# Usage:
#   ./run.sh              # normal launch
#   ./run.sh --pt         # force .pt model (skip engine)
#   ./run.sh --engine     # force .engine model (fail if not built)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$HOME/fish_venv"
EXPORT_DIR="$SCRIPT_DIR/../export"
ENGINE_PATH="$EXPORT_DIR/best.engine"
PT_PATH="$EXPORT_DIR/best.pt"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  🐟  Fish Detection & Counting — Launcher${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

# ── Parse flags ───────────────────────────────────────────────────────────────
FORCE_PT=0
FORCE_ENGINE=0
for arg in "$@"; do
    case $arg in
        --pt)     FORCE_PT=1 ;;
        --engine) FORCE_ENGINE=1 ;;
        --help|-h)
            echo "Usage: ./run.sh [--pt | --engine]"
            echo "  (no flag)   auto-select: engine if built, else .pt"
            echo "  --pt        force use of best.pt (PyTorch, slower)"
            echo "  --engine    force use of best.engine (TensorRT, faster)"
            exit 0 ;;
    esac
done

# ── Activate virtual environment ──────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo -e "  ${GREEN}✅ Venv active:${NC} $VENV_DIR"
else
    echo -e "  ${YELLOW}⚠️  No venv found at $VENV_DIR — using system Python${NC}"
fi

# ── Select model ──────────────────────────────────────────────────────────────
if [ "$FORCE_PT" -eq 1 ]; then
    if [ ! -f "$PT_PATH" ]; then
        echo -e "  ${RED}❌ best.pt not found: $PT_PATH${NC}"
        exit 1
    fi
    export FISH_MODEL_PATH="$PT_PATH"
    echo -e "  ${YELLOW}⚡ Model:${NC} best.pt (FP32, forced via --pt)"

elif [ "$FORCE_ENGINE" -eq 1 ]; then
    if [ ! -f "$ENGINE_PATH" ]; then
        echo -e "  ${RED}❌ best.engine not found: $ENGINE_PATH${NC}"
        echo -e "  ${CYAN}   Run first:  python export_model.py --format engine --half${NC}"
        exit 1
    fi
    export FISH_MODEL_PATH="$ENGINE_PATH"
    echo -e "  ${GREEN}🚀 Model:${NC} best.engine (TensorRT FP16, forced via --engine)"

else
    # Auto-select: prefer .engine if it exists
    if [ -f "$ENGINE_PATH" ]; then
        export FISH_MODEL_PATH="$ENGINE_PATH"
        echo -e "  ${GREEN}🚀 Model:${NC} best.engine (TensorRT — auto selected)"
    elif [ -f "$PT_PATH" ]; then
        export FISH_MODEL_PATH="$PT_PATH"
        echo -e "  ${YELLOW}⚡ Model:${NC} best.pt (fallback — run export_model.py to build engine)"
    else
        echo -e "  ${RED}❌ No model found in $EXPORT_DIR${NC}"
        echo -e "     Expected: best.engine  or  best.pt"
        exit 1
    fi
fi

echo -e "  ${CYAN}   Path:${NC}  $FISH_MODEL_PATH"
echo ""

# ── Set DISPLAY if not already set (needed when launching via SSH or autostart)
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo -e "  ${YELLOW}⚠️  DISPLAY not set — defaulting to :0${NC}"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo -e "  ${BOLD}▶  Starting app…${NC}"
echo ""
cd "$SCRIPT_DIR"
exec python3 main.py "$@"
