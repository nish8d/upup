#!/usr/bin/env bash
# Download and extract the Vimeo-90K triplet dataset (Xue et al., 2019).
# ~33 GB zip + ~33 GB extracted. Re-running resumes a partial download.
# Usage: scripts/download_vimeo.sh [DEST]   (default: data)
set -euo pipefail

DEST="${1:-data}"
URL="http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip"
ZIP="$DEST/vimeo_triplet.zip"
mkdir -p "$DEST"

if [[ ! -f "$ZIP" ]]; then
  free_gb=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
  if (( free_gb < 70 )); then
    echo "Need ~70 GB free in $DEST (zip + extracted), only ${free_gb} GB available." >&2
    exit 1
  fi
fi

wget -c -O "$ZIP" "$URL"
unzip -q -n "$ZIP" -d "$DEST"

if [[ ! -f "$DEST/vimeo_triplet/tri_trainlist.txt" ]]; then
  echo "Extraction finished but $DEST/vimeo_triplet/tri_trainlist.txt is missing." >&2
  exit 1
fi
echo "Dataset ready at $DEST/vimeo_triplet ($(wc -l < "$DEST/vimeo_triplet/tri_trainlist.txt") train triplets)."
echo "You can delete $ZIP to free ~33 GB."
