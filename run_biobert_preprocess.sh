#!/usr/bin/env bash
python "drive/MyDrive/Colab Notebooks/bert_relation_extraction/preprocess.py" \
--bert_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/model/Biobert/" \
--data_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/input/data/" \
--log_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/logs/" \
--main_log_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/logs/Biobert-main.log" \
--preprocess_log_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/logs/Biobert-preprocess.log" \
--output_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/checkpoint/Biobert/" \
--num_tags=4 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=128 \
--lr=2e-5 \
--other_lr=2e-4 \
--train_batch_size=32 \
--train_epochs=100 \
--eval_batch_size=32 \
--dropout_prob=0.1 \