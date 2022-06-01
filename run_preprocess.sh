#!/usr/bin/env bash
python preprocess.py \
--bert_dir="model/biobert/" \
--data_dir="input/data/" \
--log_dir="output/logs/" \
--output_dir="input/checkpoint/" \
--num_tags=4 \
--seed=123 \
--gpu_ids="-1" \
--max_seq_len=128 \
--lr=3e-5 \
--other_lr=3e-4 \
--train_batch_size=32 \
--train_epochs=100 \
--eval_batch_size=32 \
--dropout_prob=0.3 \
