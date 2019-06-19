#!/usr/bin/env bash

command="$0 $@"
basedir=$(dirname "$0")
main_prog="$basedir/../main.py"
rend_prog="$basedir/../render.py"

function base_usage {
  echo "Usage: $0 [-c checkpoint_dir] [-f] [-g gpu] [-h|--help]"
}

function usage {
  base_usage
  python3 "$main_prog" -h
}

if [[ $# -eq 0 ]]; then
  base_usage
  exit 0
fi

params=
GPUID=0
cptd=
overwrite=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c)
      params="$params --checkpoint_dir $2"
      cptd="$2"
      shift
      ;;
    -f)
      overwrite=1
      ;;
    -g)
      GPUID="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      params="$params $1"
      ;;
  esac
  shift
done

# requires a checkpoint
if [[ -z "$cptd" ]] || [[ ! -d "$cptd" ]]; then
  echo "Either missing or invalid checkpoint directory"
  exit 1
fi

# save command in case we crash
echo "$command" >> "$basedir"/evals.log

# output info
echo "Using gpu $GPUID"
echo "Parameters: $params"

# copy evaluation to experiments
src_dir="$cptd/eval"
exp_name=$(basename "$cptd")
trg_base_dir="$basedir/results"
trg_prog_dir="$trg_base_dir/programs/${exp_name}"

if [[ ! -d "$trg_prog_dir" ]]; then
  echo "Creating experiment in $trg_prog_dir"
  mkdir -p "$trg_prog_dir"
elif [[ $overwrite -eq 0 ]]; then
  echo "Experiment already in $trg_prog_dir"
  echo "Use -f to overwrite it"
  exit 2
else
  echo "Overwriting experiment in $trg_prog_dir"
fi

# run actual script
# @see https://github.com/tensorflow/tensorflow/issues/379
CUDA_VISIBLE_DEVICES="$GPUID" python3 "$main_prog" $params --notraining --batch_size=1
cp -R "$src_dir"/* "$trg_prog_dir"

# create renderings
trg_rend_dir="$trg_base_dir/renderings/${exp_name}"

if [[ ! -d "$trg_rend_dir" ]]; then
  echo "Creating renderings in $trg_rend_dir"
  mkdir -p "$trg_rend_dir"
elif [[ $overwrite -eq 0 ]]; then
  echo "Experiment already has renderings in $trg_rend_dir"
  echo "Use -f to overwrite it"
  exit 3
else
  echo "Overwriting renderings in $trg_rend_dir"
fi

CUDA_VISIBLE_DEVICES="$GPUID" python3 "$rend_prog" --output_dir="$trg_rend_dir" "$src_dir"/*.png
