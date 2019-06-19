#!/usr/bin/env bash

size="$1"

if [[ ! "$size" =~ ^[0-9]+$ ]]; then
  echo "Must provide size arguments"
  exit 1
fi

subsize="$2"

if [[ ! "$subsize" =~ ^[0-9]+$ ]]; then
  echo "Must provide subsize argument"
  exit 3
fi

basedir=$(dirname "$0")
datadir="$basedir/${size}_${subsize}"
if [[ -d "$datadir" ]]; then
  echo "Directory $datadir already exists"
  exit 2
fi

echo "Creating subset dataset of size real=$size synt=$subsize"
mkdir "$datadir"

basedataset_dir="$basedir/../dataset"
for f in "$basedataset_dir"/*; do
  ln -s $(realpath "$f") "$datadir/" # create link to original dataset
done

declare -A sizes
sizes=([real]=$size [synt]=$subsize)

for datatype in real synt; do
  ssize=${sizes[$datatype]}
  src_list="$basedataset_dir/train_${datatype}.txt"
  trg_list="$datadir/train_${datatype}.txt"
  # replace real training file
  rm "$trg_list"
  cat "$src_list" | sort -R | head -n "$ssize" > "$trg_list"

  echo "Categories (full) for $datatype:"
  cat "$src_list" | cut -d'_' -f1 | sort | uniq -c
  echo "Categories ($ssize) for $datatype:"
  cat "$trg_list" | cut -d'_' -f1 | sort | uniq -c
done
