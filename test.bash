#!/usr/bin/env bash
# sanity_test.sh — runs the full single-patcher pipeline end to end
# Usage: bash sanity_test.sh [config]
 
set -e
 
CONFIG=${1:-configs/tiny.yaml}
 
echo "=== Using config: $CONFIG ==="
 
echo ""
echo "[1/4] Preparing corpus data..."
python scripts/prepare_data.py --config "$CONFIG"
 
echo ""
echo "[2/4] Training patcher..."
python scripts/train_patcher.py --config "$CONFIG"
 
echo ""
echo "[3/4] Caching patcher hidden states..."
python scripts/prepare_data_patcher2.py \
  --config "$CONFIG" \
  --patcher-checkpoint outputs/patcher/best.pt
 
echo ""
echo "[4/4] Training trunk..."
python scripts/train_tiny.py --config "$CONFIG"
 
echo ""
echo "=== Sanity test complete ==="
 
