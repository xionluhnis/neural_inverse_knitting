#!/usr/bin/env bash

size="$1"

if [[ ! "$size" =~ ^[0-9]+$ ]]; then
  echo "Must provide size arguments"
  exit 1
fi

basedir=$(dirname "$0")
datadir="$basedir/$size"
if [[ -d "$datadir" ]]; then
  echo "Directory $datadir already exists"
  exit 2
fi

echo "Creating subset dataset of size $size"
mkdir "$datadir"

basedataset_dir="$basedir/../dataset"
for f in "$basedataset_dir"/*; do
  ln -s $(realpath "$f") "$datadir/" # create link to original dataset
done

src_list="$basedataset_dir/train_real.txt"
trg_list="$datadir/train_real.txt"
# replace real training file
rm "$trg_list"
cat "$src_list" | sort -R | head -n "$size" > "$trg_list"

echo "Categories (full):"
cat "$src_list" | cut -d'_' -f1 | sort | uniq -c
echo "Categories ($size):"
cat "$trg_list" | cut -d'_' -f1 | sort | uniq -c
