#!/usr/bin/env bash

basedir=$(dirname "$0")
[[ -z "$GPUID" ]] && GPUID=0

[[ ! -d "$basedir/programs/gt" ]] && mkdir -p "$basedir/programs/gt"
[[ ! -d "$basedir/renderings/gt" ]] && mkdir -p "$basedir/renderings/gt"

datadir="$basedir/../../dataset"
if [[ ! -d "$datadir" ]]; then
  echo "You must download the dataset first!"
  exit 1
fi

echo "Copying ground truth instructions"
cat "$datadir/test_real.txt" | while read -r name; do
  echo "$name"
  cp "$datadir/instruction/${name}.png" "$basedir/programs/gt/"
  tput cuu1 && tput el
done
echo "Done"

echo "Rendering ground truth for perceptual metrics"
CUDA_VISIBLE_DEVICES="$GPUID" python3 "$basedir/../../render.py" --output_dir="$basedir/renderings/gt" "$basedir/programs/gt/"*.png
