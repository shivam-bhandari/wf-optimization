#!/bin/bash

# Prepare data and images for the web dashboard
set -euo pipefail
LATEST_DIR=$(ls -d results/*/ 2>/dev/null | sort -V | tail -n 1)
WEB_DATA_DIR="web/data"
WEB_IMAGES_DIR="web/images"

if [[ -z "$LATEST_DIR" || ! -d "$LATEST_DIR" ]]; then
  echo "[ERROR] No results directories found. Please run the benchmark pipeline first."
  exit 1
fi
WEB_RESULTS_FILE="$LATEST_DIR/web_results.json"
if [[ ! -f "$WEB_RESULTS_FILE" ]]; then
  echo "[ERROR] $WEB_RESULTS_FILE not found. Run benchmarks to generate it."
  exit 1
fi
mkdir -p "$WEB_DATA_DIR"
cp "$WEB_RESULTS_FILE" "$WEB_DATA_DIR/web_results.json"

# Copy images
title=""
if [[ -d "$LATEST_DIR/visualizations" ]]; then
  mkdir -p "$WEB_IMAGES_DIR"
  pngs=("$LATEST_DIR/visualizations"/*.png)
  if [[ -e ${pngs[0]} ]]; then
    cp "$LATEST_DIR/visualizations"/*.png "$WEB_IMAGES_DIR/"
    title="Copied $(ls -1 "$LATEST_DIR/visualizations"/*.png | wc -l) PNG charts."
  else
    title="No PNG files found in $LATEST_DIR/visualizations/."
  fi
else
  title="No visualizations directory found in $LATEST_DIR."
fi
# Write/update last_updated.json
NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "{\"last_updated\": \"$NOW\"}" > "$WEB_DATA_DIR/last_updated.json"

# Summary output
echo "[SUCCESS] Copied web_results.json to $WEB_DATA_DIR/."
echo "[SUCCESS] $title"
echo "[SUCCESS] Updated last_updated.json."
echo "Web data ready for local serving in web/."
