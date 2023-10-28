#!/bin/bash
set -e

RUSTFLAGS='-C target-cpu=native' cargo build --release -p parse
cp target/release/libparse.so ./

TRAIN_ID="$(date -u '+%F-%H%M%S')-$1"

NETS=`mktemp -d`

python3 trainer/main.py --nndir "$NETS" --data "datasets/$1" --scale 1016 --save-epochs 1 \
    --lr 0.001 --epochs 20 --lr-drop 15 --wdl 1.0 \
    | tee /dev/stderr >"$NETS"/log

pushd "$NETS" >/dev/null
tar cf networks.tar *
popd >/dev/null
zstd "$NETS/networks.tar" -o "$TRAIN_ID.tar.zst"
echo "$TRAIN_ID"
rm -rf "$NETS"
