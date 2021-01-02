#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

./train.py --arch="mynet" \
    --start-epoch=1 \
    --loss=wpdc \
    --snapshot="mynet" \
    --param-fp-train='../train.configs/param_all_norm.pkl' \
    --param-fp-val='../train.configs/param_all_norm_val.pkl' \
    --warmup=5 \
    --opt-style=resample \
    --resample-num=132 \
    --batch-size=512 \
    --base-lr=0.02 \
    --epochs=50 \
    --milestones=30,40 \
    --print-freq=50 \
    --devices-id=1 \
    --workers=0 \
    --filelists-train="../train.configs/train_aug_120x120.list.train" \
    --filelists-val="../train.configs/train_aug_120x120.list.val" \
    --root="/home/shishang/Desktop/datasets/ALFW/train_aug_120x120" \
    --log-file="${LOG_FILE}"