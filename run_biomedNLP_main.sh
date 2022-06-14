#!/usr/bin/env bash
python "drive/MyDrive/Colab Notebooks/bert_relation_extraction/main.py" \
--bert_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/model/BiomedNLP-PubMedBERT/" \
--data_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/input/data/" \
--log_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/logs/" \
--main_log_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/logs/BiomedNLP-PubMedBERT-main.log" \
--preprocess_log_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/logs/BiomedNLP-PubMedBERT-preprocess.log" \
--output_dir="drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/checkpoint/BiomedNLP-PubMedBERT/" \
--num_tags=4 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=128 \
--lr=1e-5 \
--other_lr=1e-4 \
--train_batch_size=16 \
--train_epochs=100 \
--eval_batch_size=16 \
--dropout_prob=0.1 \