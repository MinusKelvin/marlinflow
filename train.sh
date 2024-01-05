#!/bin/bash
set -e

if [ -z "$VIRTUAL_ENV" ]; then
    echo "not in venv"
    exit 1
fi

RUSTFLAGS='-C target-cpu=native' cargo build --release -p parse
cp target/release/libparse.so .

python3 trainer/main.py --data "$1" --scale 1016 --nndir trainings \
    --lr 0.01 --epochs 15 --lr-drop 8 --lr-drop 12 --wdl 1
