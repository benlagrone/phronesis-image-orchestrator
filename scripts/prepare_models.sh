#!/usr/bin/env bash
set -euo pipefail

# Prepare Stable Diffusion model folders and files inside ./models
# - Reads folder list from models1.txt
# - Reads model files (filenames or URLs) from models2.txt
# - Copies files from a local source dir if provided, or downloads URLs
#
# Usage:
#   MODELS_SRC_DIR=/path/to/your/model/files scripts/prepare_models.sh
# or
#   scripts/prepare_models.sh /path/to/your/model/files
#
# Notes:
# - If a line in models2.txt is an http(s) URL, it will be downloaded.
# - If it is a filename, we look under MODELS_SRC_DIR for a matching file and copy it.
# - Missing files are reported at the end.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODELS_DIR="${MODELS_DIR:-models}"
STABLE_DIR="$MODELS_DIR/Stable-diffusion"

SRC_DIR_FROM_ARG="${1-}"
MODELS_SRC_DIR="${MODELS_SRC_DIR-${SRC_DIR_FROM_ARG}}"

MODELS1_FILE="models1.txt"
MODELS2_FILE="models2.txt"

if [[ ! -f "$MODELS1_FILE" ]]; then
  echo "ERROR: $MODELS1_FILE not found in repo root: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -f "$MODELS2_FILE" ]]; then
  echo "ERROR: $MODELS2_FILE not found in repo root: $REPO_ROOT" >&2
  exit 1
fi

echo "Preparing model directories under $MODELS_DIR ..."
mkdir -p "$MODELS_DIR"

trim_line() {
  local s="$1"
  # drop comments
  s="${s%%#*}"
  # drop trailing CR
  s="${s%$'\r'}"
  # trim leading/trailing whitespace
  s="$(printf '%s' "$s" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  printf '%s' "$s"
}

# Create subdirectories listed in models1.txt
while IFS= read -r raw; do
  name="$(trim_line "$raw")"
  [[ -z "$name" ]] && continue
  mkdir -p "$MODELS_DIR/$name"
done < "$MODELS1_FILE"

# Always ensure Stable-diffusion exists
mkdir -p "$STABLE_DIR"

echo "\nPlacing model files into $STABLE_DIR ..."
missing=()

download() {
  url="$1"
  base="${url##*/}"
  dest="$STABLE_DIR/$base"
  echo "  - Downloading $base"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$dest" "$url" || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$url" || return 1
  else
    echo "    WARN: Neither curl nor wget is available to download $url" >&2
    return 1
  fi
}

copy_from_src() {
  fname="$1"
  src_dir="$2"
  # try exact filename
  if [[ -n "$src_dir" && -f "$src_dir/$fname" ]]; then
    echo "  - Copying $fname from $src_dir"
    cp -f "$src_dir/$fname" "$STABLE_DIR/"
    return 0
  fi
  # try case-insensitive or partial matches in source dir
  if [[ -n "$src_dir" && -d "$src_dir" ]]; then
    match=$(find "$src_dir" -maxdepth 1 -type f -iname "$fname" -print -quit 2>/dev/null || true)
    if [[ -n "$match" ]]; then
      base="$(basename "$match")"
      echo "  - Copying $base from $src_dir"
      cp -f "$match" "$STABLE_DIR/"
      return 0
    fi
  fi
  return 1
}

while IFS= read -r raw; do
  item="$(trim_line "$raw")"
  [[ -z "$item" ]] && continue

  if [[ "$item" =~ ^https?:// ]]; then
    if ! download "$item"; then
      missing+=("$item")
    fi
  else
    if ! copy_from_src "$item" "${MODELS_SRC_DIR-}"; then
      echo "  - MISSING: $item"
      missing+=("$item")
    fi
  fi
done < "$MODELS2_FILE"

echo "\nDone. Directory layout:"
find "$MODELS_DIR" -maxdepth 2 -type d -print | sed 's,^,  ,'

if ((${#missing[@]} > 0)); then
  echo "\nThe following items were not found or failed to download:"
  for m in "${missing[@]}"; do echo "  - $m"; done
  echo "\nTips:"
  echo "  - Set MODELS_SRC_DIR to the folder where your model files live"
  echo "  - Or replace filenames in models2.txt with direct URLs to download"
  exit 2
fi

echo "\nAll models present in $STABLE_DIR."
