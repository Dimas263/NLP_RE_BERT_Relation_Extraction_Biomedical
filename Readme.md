# <img src="https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/undefined/external-big-data-smart-technology-flaticons-lineal-color-flat-icons-2.png"/> **NLP Research**
# **Relation Extraction in Biomedical using Bert-LSTM-CRF model and pytorch**
## <img src="https://img.icons8.com/external-fauzidea-flat-fauzidea/64/undefined/external-man-avatar-avatar-fauzidea-flat-fauzidea.png"/> **`Dimas Dwi Putra`**
<img src="https://img.icons8.com/metro/26/undefined/chevron-right.png"> NLP Research - Bert Relation Extraction in Biomedical.<br>

<img src="https://img.icons8.com/color/48/undefined/python--v1.png"/> [Notebook 1.ipynb](BiomedNLP_PubMedBERT_Notebook.ipynb)<br>`Created using BiomedNLP-PubMedBERT Pre-Trained Model`<br>

<img src="https://img.icons8.com/color/48/undefined/python--v1.png"/> [Notebook 2.ipynb](Biobert_Notebook.ipynb)<br>`Created using Biobert Pre-Trained Model`

# <img src="https://img.icons8.com/color/48/undefined/1-circle--v1.png"/> Config

## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> BioNLP
```yaml
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
```

## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> Biobert
```yaml
#!/usr/bin/env bash
python "drive/MyDrive/Colab Notebooks/bert_relation_extraction/main.py" \
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
```

# <img src="https://img.icons8.com/color/48/undefined/2-circle--v1.png"/> Dataset 
# <img src="https://img.icons8.com/color/48/undefined/folder-invoices--v1.png"/> [View Directory](input/data/)

<img src="input/data/bar_chart.png" width="400">

## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> Dictionary
```json
{"Cause_of_disease": 0, "Treatment_of_disease": 1, "Negative": 2, "Association": 3}
```
## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> Data Preprocessing

<img src="input/data/data-preprocessing.png" width="600">

## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> Example
```yaml
id_relation	<e1start>entity1</e1end>	<e2start>entity2</e2end>	sentence	start_entity1	end_entity1	start_entity2	end_entity2
```
## - Training Set
```yaml
2	The evidence for <e1start> soybean <e1end> products as <e2start> cancer <e2end> preventive agents.  	17	42	55	79
1	[Mortality trends in <e2start> cancer <e2end> attributable to <e1start> tobacco <e1end> in Mexico].  	62	87	21	45
3	<e1start> Areca <e1end> nut chewing has a significant association with <e2start> systemic inflammation <e2end>.	0	23	71	110
...
```
## - Testing Set
```yaml
1	Its effect on <e1start> digitalis <e1end>-caused <e2start> atrial arrhythmias <e2end> is unknown. 	14	41	49	85
0	However, the growth rate of <e2start> tumors <e2end> was not markedly inhibited by <e1start> garlic <e1end>. 	83	107	28	52
1	<e1start> Tobacco <e1end>-related <e2start> cancers <e2end> in Madras, India.  	0	25	34	59
...
```

# <img src="https://img.icons8.com/color/48/undefined/3-circle--v1.png"/> Model input 
# <img src="https://img.icons8.com/color/48/undefined/folder-invoices--v1.png"/>[View Directory](model/)

## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> BioNLP

`git clone https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`

## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> Biobert

`git clone https://huggingface.co/dmis-lab/biobert-v1.1`


## <img src="https://img.icons8.com/color/48/undefined/4-circle--v1.png"/> Preprocessing
```yaml
2022-06-14 20:24:44,227 - INFO - preprocess.py - <module> - 180 - {'output_dir': 'drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/checkpoint/BiomedNLP-2022-06-14 20:24:47,180 - INFO - preprocess.py - convert_bert_example - 95 - *** train_example-0 ***
2022-06-14 20:24:47,181 - INFO - preprocess.py - convert_bert_example - 96 - text: [CLS] [UNK] o w e v e r, [UNK] m o r e [UNK] s t u d i e s [UNK] n e e d [UNK] t o [UNK] f u r t h e r [UNK] e x p l o r e [UNK] t h e [UNK] r o l e s [UNK] o f [UNK] v i t e x [UNK] a g n u s [UNK] c a s t u s [UNK] i n [UNK] f r a c t u r e [UNK] r e p a i r [UNK] p r o c e s s e s. [UNK] [SEP]
2022-06-14 20:24:47,181 - INFO - preprocess.py - convert_bert_example - 97 - token_ids: [2, 1, 57, 65, 47, 64, 47, 60, 16, 1, 55, 57, 60, 47, 1, 61, 62, 63, 46, 51, 47, 61, 1, 56, 47, 47, 46, 1, 62, 57, 1, 48, 63, 60, 62, 50, 47, 60, 1, 47, 66, 58, 54, 57, 60, 47, 1, 62, 50, 47, 1, 60, 57, 54, 47, 61, 1, 57, 48, 1, 64, 51, 62, 47, 66, 1, 43, 49, 56, 63, 61, 1, 45, 43, 61, 62, 63, 61, 1, 51, 56, 1, 48, 60, 43, 45, 62, 63, 60, 47, 1, 60, 47, 58, 43, 51, 60, 1, 58, 60, 57, 45, 47, 61, 61, 47, 61, 18, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2022-06-14 20:24:47,181 - INFO - preprocess.py - convert_bert_example - 98 - attention_masks: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2022-06-14 20:24:47,181 - INFO - preprocess.py - convert_bert_example - 99 - token_type_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2022-06-14 20:24:47,182 - INFO - preprocess.py - convert_bert_example - 100 - labels: 0
2022-06-14 20:24:47,182 - INFO - preprocess.py - convert_bert_example - 101 - ids: [60, 78, 82, 90]
. . . 
```
[LOAD MORE..](output/logs)

# <img src="https://img.icons8.com/color/48/undefined/5-circle--v1.png"/> Output

## Visualization

## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> BioNLP

<img src="output/visualization/BiomedNLP-PubMedBERT-train.png" width="300"><img src="output/visualization/BiomedNLP-PubMedBERT-dev.png" width="300">

## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> Biobert

<img src="output/visualization/biobert-train.png" width="300"><img src="output/visualization/biobert-dev.png" width="300">

[LOAD MORE.. ](output/visualization)

## Train, Test, Predict

## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> BioNLP

- Train
```yaml
2022-06-14 20:35:06,300 - INFO - main.py - train - 86 - 【train】 epoch：48 step:580/1200 loss：0.025188
2022-06-14 20:35:06,651 - INFO - main.py - train - 92 - 【dev】 loss：2.567647 accuracy：0.7872 micro_f1：0.7872 macro_f1：0.8343
2022-06-14 20:35:06,652 - INFO - main.py - train - 94 - ------------>Save best model
...
```
- Test
```yaml
2022-06-14 20:40:56,848 - INFO - main.py - <module> - 247 - ======== Calculate Testing========
2022-06-14 20:40:59,872 - INFO - main.py - <module> - 251 - 【test】 loss：2.567647 accuracy：0.7872 micro_f1：0.7872 macro_f1：0.8343
```
<center>

```
                        precision    recall   per-class   support
                                              f1-scores

            Negative       0.83      0.67      0.74        15
    Cause_of_disease       0.69      0.92      0.79        12
Treatment_of_disease       0.83      0.79      0.81        19
         Association       1.00      1.00      1.00         1
```

```
                          precision    recall   Average     support
                                               f1-scores

            accuracy                           0.79        47
           macro avg       0.84      0.84      0.83        47
        weighted avg       0.80      0.79      0.79        47
```
</center>

- Predict
```yaml
2022-06-14 20:41:18,016 - INFO - predict.py - <module> - 200 - ======== Prediction ========
2022-06-14 20:41:27,231 - INFO - predict.py - <module> - 213 - OBJECTIVE: To study the role of pecan tree pollen in the development of allergy . 
2022-06-14 20:41:29,887 - INFO - predict.py - <module> - 215 - predict labels：Negative
2022-06-14 20:41:29,887 - INFO - predict.py - <module> - 216 - true label：Negative
2022-06-14 20:41:29,888 - INFO - predict.py - <module> - 217 - ==========================
2022-06-14 20:42:00,013 - INFO - predict.py - <module> - 213 - A lipid-soluble red ginseng extract inhibits the growth of human lung tumor xenografts in nude mice. 
2022-06-14 20:42:02,499 - INFO - predict.py - <module> - 215 - predict labels：Treatment_of_disease
2022-06-14 20:42:02,500 - INFO - predict.py - <module> - 216 - true label：Treatment_of_disease
2022-06-14 20:42:02,500 - INFO - predict.py - <module> - 217 - ==========================
...
```

## <img src="https://img.icons8.com/tiny-color/16/undefined/experimental-right-tiny-color.png"/> Biobert

- Train
```yaml
2022-06-14 20:29:44,894 - INFO - main.py - train - 86 - 【train】 epoch：49 step:299/600 loss：0.011506
2022-06-14 20:29:45,223 - INFO - main.py - train - 92 - 【dev】 loss：1.801206 accuracy：0.7447 micro_f1：0.7447 macro_f1：0.8104
2022-06-14 20:29:45,224 - INFO - main.py - train - 94 - ------------>Save best model
...
```
- Test
```yaml
2022-06-14 20:33:26,034 - INFO - main.py - <module> - 247 - ======== Calculate Testing========
2022-06-14 20:33:28,844 - INFO - main.py - <module> - 251 - 【test】 loss：1.801206 accuracy：0.7447 micro_f1：0.7447 macro_f1：0.8104
```
<center>

```
                        precision    recall   per-class   support
                                    f1-scores

            Negative       0.61       0.73      0.67        15
    Cause_of_disease       0.90       0.75      0.82        12
Treatment_of_disease       0.78       0.74      0.76        19
         Association       1.00       1.00      1.00         1
```

```
                          precision    recall   Average     support
                                    f1-scores

            accuracy                              0.74        47
           macro avg       0.82         0.81      0.81        47
        weighted avg       0.76         0.74      0.75        47
```
</center>

- Predict
```yaml
2022-06-14 20:41:18,016 - INFO - predict.py - <module> - 200 - ======== Prediction ========
2022-06-14 20:41:27,231 - INFO - predict.py - <module> - 213 - OBJECTIVE: To study the role of pecan tree pollen in the development of allergy . 
2022-06-14 20:41:29,887 - INFO - predict.py - <module> - 215 - predict labels：Negative
2022-06-14 20:41:29,887 - INFO - predict.py - <module> - 216 - true label：Negative
2022-06-14 20:41:29,888 - INFO - predict.py - <module> - 217 - ==========================
2022-06-14 20:42:00,013 - INFO - predict.py - <module> - 213 - A lipid-soluble red ginseng extract inhibits the growth of human lung tumor xenografts in nude mice. 
2022-06-14 20:42:02,499 - INFO - predict.py - <module> - 215 - predict labels：Treatment_of_disease
2022-06-14 20:42:02,500 - INFO - predict.py - <module> - 216 - true label：Treatment_of_disease
2022-06-14 20:42:02,500 - INFO - predict.py - <module> - 217 - ==========================
...
```
[LOAD MORE.. ](output/logs)

# <img src="https://img.icons8.com/color/48/undefined/6-circle--v1.png"/> Model output
<img src="https://img.icons8.com/external-sbts2018-flat-sbts2018/48/undefined/external-model-basic-ui-elements-2.4-sbts2018-flat-sbts2018.png"/> [`1.3G Jun 14 20:35 best.pt`](https://drive.google.com/drive/folders/1_xUN_FlX9-4kt_CYCnf-gHwJh7G6k6qg?usp=sharing)

`Best.pt`
model created by pytorch after Train and Validation 

# <img src="https://img.icons8.com/color/48/undefined/7-circle--v1.png"/> Summary
<img src="https://img.icons8.com/color/48/undefined/microsoft-excel-2019--v1.png"/> [**Relation Extraction Report**](https://docs.google.com/spreadsheets/d/1Q6JgAC5rDz4EM00BuAlEaraC0tm3oZoYqBu_F01md7s/edit?usp=sharing)
