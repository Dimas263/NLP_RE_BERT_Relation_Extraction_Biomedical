#!/usr/bin/env bash
python "predict.py" \
--bert_dir="model/BiomedNLP-PubMedBERT/" \
--data_dir="input/data/" \
--log_dir="output/logs/" \
--output_dir="output/checkpoint/" \
--num_tags=4 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=128 \
--lr=1e-5 \
--other_lr=1e-4 \
--train_batch_size=64 \
--train_epochs=500 \
--eval_batch_size=64 \
--dropout_prob=0.1 \