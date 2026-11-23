#!/usr/bin/env bash
# fusionesp_setup.sh
# Downloads two .tar archives, extracts them,
# and removes each archive after extracting.

set -euo pipefail

# --- Config ---
DATA_URL="https://huggingface.co/datasets/lizmahood/cyp_pred_repos/resolve/main/ezspec_data_dir.tar.zst"
CKPT_URL="https://huggingface.co/datasets/lizmahood/cyp_pred_repos/resolve/main/ezspec_model_ckpts.tar.zst"

# Filenames to save as (clean names without query string)
DATA_TZST="ezspec_data_dir.tar.zst"
CKPT_TZST="ezspec_model_ckpts.tar.zst"

# --- Helpers ---
have() { command -v "$1" >/dev/null 2>&1; }

dl() {
  local url="$1" out="$2"
  echo "==> Downloading: $url"
  if have curl; then
    curl -fL --retry 5 --retry-connrefused --retry-delay 2 -o "$out" "$url"
  elif have wget; then
    wget --tries=5 --retry-connrefused -O "$out" "$url"
  else
    echo "Error: neither curl nor wget is installed." >&2
    exit 1
  fi
}

extract_tzst_into_parent_dir() {
  # Extracts archive next to itself (into its containing directory).
  local tzst="$1"
  local parent
  parent="$(dirname "$tzst")"
  echo "==> Extracting: $tzst"

  if tar --help 2>/dev/null | grep -q -- '--zstd'; then
    tar --zstd -xf "$tzst" -C "$parent"
  elif have unzstd; then
    unzstd -c "$tzst" | tar -xf - -C "$parent"
  elif have zstd; then
    zstd -dc "$tzst" | tar -xf - -C "$parent"
  else
    echo "Error: need either tar with --zstd support, or zstd/unzstd installed." >&2
    exit 1
  fi
}

# --- Main ---
# Destination directory (default: current working dir)
DEST_DIR="${1:-.}"
mkdir -p "$DEST_DIR"
DEST_DIR="$(cd "$DEST_DIR" && pwd -P)"

echo "Working directory: $DEST_DIR"
cd "$DEST_DIR"

# 1 & 2) Download the two archives (skip if present and non-empty)
if [[ ! -s "$DATA_TZST" ]]; then
  dl "$DATA_URL" "$DATA_TZST"
else
  echo "==> Skipping download; file exists: $DATA_TZST"
fi

if [[ ! -s "$CKPT_TZST" ]]; then
  dl "$CKPT_URL" "$CKPT_TZST"
else
  echo "==> Skipping download; file exists: $CKPT_TZST"
fi

# 3) Extract both top-level archives into DEST_DIR, then delete them
extract_tzst_into_parent_dir "$DATA_TZST"
rm -f -- "$DATA_TZST"

extract_tzst_into_parent_dir "$CKPT_TZST"
rm -f -- "$CKPT_TZST"

echo "Done!"
