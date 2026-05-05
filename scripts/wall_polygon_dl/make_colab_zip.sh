#!/usr/bin/env bash
# Pack code + cached dataset for Colab upload.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$HERE/../../../_workspace/sprint_57_neural_polygon" && pwd)"
CACHE="${1:-$WORKSPACE/cache_v1}"
OUT="${2:-$WORKSPACE/wall_polygon_dl_colab.zip}"

if [[ ! -d "$CACHE" ]]; then
  echo "ERROR: cache dir not found: $CACHE" >&2
  echo "Run: python precompute_dataset.py --n 1000 --out-dir $CACHE" >&2
  exit 1
fi

CODE_FILES=(
  "data_generator.py"
  "dataset.py"
  "model.py"
  "train.py"
)

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "[stage] copy code → $STAGE/"
for f in "${CODE_FILES[@]}"; do
  cp "$HERE/$f" "$STAGE/$f"
done

echo "[stage] copy cache → $STAGE/cache_v1/"
mkdir -p "$STAGE/cache_v1"
cp "$CACHE"/*.npz "$STAGE/cache_v1/"

CACHE_COUNT=$(find "$STAGE/cache_v1" -name "*.npz" | wc -l | tr -d ' ')
echo "[stage] cached samples: $CACHE_COUNT"

rm -f "$OUT"
echo "[zip] writing $OUT"
( cd "$STAGE" && zip -r -q "$OUT" . )
SIZE=$(du -h "$OUT" | cut -f1)
echo "[done] $OUT ($SIZE, $CACHE_COUNT samples + ${#CODE_FILES[@]} files)"
