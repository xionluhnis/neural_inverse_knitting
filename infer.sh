#!/usr/bin/env bash

command="$0 $@"

function base_usage {
  echo "Usage: $0 [-c checkpoint_dir] [-g gpu] [-o output_dir] [-h|--help] [-v] img1 [img2 ...]"
}

function usage {
  base_usage
  python3 main.py -h
}

if [[ $# -eq 0 ]]; then
  base_usage
  exit 0
fi

params=
GPUID=0
cptd=
output_dir=./
inputs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c)
      params="$params --checkpoint_dir $2"
      cptd="$2"
      shift
      ;;
    -g)
      GPUID="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -o)
      output_dir="$2"
      if [[ ! -d "$output_dir" ]]; then
        mkdir "$output_dir"
      fi
      shift
      ;;
    -v)
      params="$params --params show_confidence=1"
      ;;
    *)
      if [[ -f "$1" ]]; then
        inputs+=( "$1" )
      else
        params="$params $1"
      fi
      ;;
  esac
  shift
done

if [[ ${#inputs[@]} -lt 1 ]]; then
  echo "Needs image inputs"
  exit 1
fi

# requires a checkpoint
if [[ -z "$cptd" ]] || [[ ! -d "$cptd" ]]; then
  echo "Either missing or invalid checkpoint directory"
  exit 1
fi

# save command in case we crash
if [[ ! -d checkpoint ]]; then
  mkdir checkpoint
fi
echo "$command" >> checkpoint/infer.log

# output info
echo "Using gpu $GPUID"
echo "Parameters: $params"
echo "Files(${#inputs[@]}): ${inputs[@]}"
# create evaluation directories
tmpdir=$(mktemp -d)
realdir="$tmpdir/real/160x160/gray"
mkdir -p "$realdir"
instdir="$tmpdir/instruction"
mkdir "$instdir"
list="$tmpdir/test_real.txt"
i=0
for img in "${inputs[@]}"; do
  name="input-$(basename "${img%.jpg}")"
  convert "$img" -fx '(r+g+b)/3' -colorspace Gray "$realdir/${name}.jpg"
  convert -size 20x20 xc:black -alpha off "$instdir/${name}.png"
  echo "$name" >> "$list"
  ((++i))
done

# copy evaluation to output directory
src_dir="$cptd/eval"

# run actual script
# @see https://github.com/tensorflow/tensorflow/issues/379
CUDA_VISIBLE_DEVICES="$GPUID" python3 main.py $params --notraining --batch_size=1 --dataset="$tmpdir"

# copy output
echo "Tempdir $tmpdir"
i=0
for img in "${inputs[@]}"; do
  name=$(basename "${img%.jpg}")
  input="input-$(basename "${img%.jpg}")"
  cp "$src_dir/${input}.png" "$output_dir/${name}.png"
  ((++i))
done

