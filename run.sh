#!/usr/bin/env bash

command="$0 $@"

function usage {
  echo "Usage: $0 [-c checkpoint_dir] [-m min_memory] [-g gpu] [-h|--help] [tf_params]"
  python3 main.py -h
}

min_memory=6000
params=

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c)
      params="$params --checkpoint_dir $2"
      shift
      ;;
    -g)
      GPUID="$2"
      shift
      ;;
    -m)
      min_memory="$2"
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

if [[ -z "$GPUID" ]]; then
  GPUID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' "{ if(\$2 > ${min_memory}) print \$1 }" | head -n 1)
  [[ -z "$GPUID" ]] && echo "No gpu available with $min_memory GB of GPU memory" && exit 1
fi

# save command in case we crash
if [[ ! -d checkpoint ]]; then
  mkdir checkpoint
fi
echo "$command" >> checkpoint/commands.log

# output info
echo "Using gpu $GPUID"
echo "Parameters: $params"

# run actual script
CUDA_VISIBLE_DEVICES="$GPUID" python3 main.py $params
