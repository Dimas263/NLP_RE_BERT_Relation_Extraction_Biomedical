# NLP - Relation Extraction Biobert
NLP Research - Bert Relation Extraction in Biomedical.

<img src="input/data/UML-RE.drawio.png" width="600">

# Config
```yaml
#!/usr/bin/env bash
python main.py \
--bert_dir="model/biobert/" \
--data_dir="input/data/" \
--log_dir="output/logs/" \
--output_dir="output/checkpoint/" \
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
```

# Dataset
[View Directory](input/data/)

<img src="input/data/bar_chart.png" width="400">

Dictionary
```json
{"Negative": 0, "Cause_of_disease": 1, "Treatment_of_disease": 2, "Association": 3}
```
Data Preprocessing

<img src="input/data/data-preprocessing.png" width="600">

example
```yaml
id_relation	<e1start>entity1</e1end>	<e2start>entity2</e2end>	sentence	start_entity1	end_entity1	start_entity2	end_entity2
```
- Training Set
```yaml
2	The evidence for <e1start> soybean <e1end> products as <e2start> cancer <e2end> preventive agents.  	17	42	55	79
1	[Mortality trends in <e2start> cancer <e2end> attributable to <e1start> tobacco <e1end> in Mexico].  	62	87	21	45
3	<e1start> Areca <e1end> nut chewing has a significant association with <e2start> systemic inflammation <e2end>.	0	23	71	110
```
- Testing Set
```yaml
1	Its effect on <e1start> digitalis <e1end>-caused <e2start> atrial arrhythmias <e2end> is unknown. 	14	41	49	85
0	However, the growth rate of <e2start> tumors <e2end> was not markedly inhibited by <e1start> garlic <e1end>. 	83	107	28	52
1	<e1start> Tobacco <e1end>-related <e2start> cancers <e2end> in Madras, India.  	0	25	34	59
```

# Model input
Biobert<br>[View Directory](model/)
```yaml
config.json
flax_model.msgpack
pytorch_model.bin
special_tokens_map.json
tokenizer_config.json
vocab.txt
```

# output
<img src="output/csv/train_chart_uji_1.png" width="400">

- Train
```yaml
2022-06-01 22:46:09,282 - INFO - main.py - train - 87 - 【train】 epoch：49 step:99/200 loss：0.005021
2022-06-01 22:47:01,244 - INFO - main.py - train - 93 - 【dev】 loss：2.787870 accuracy：0.3636 micro_f1：0.3636 macro_f1：0.4325
2022-06-01 22:47:01,276 - INFO - main.py - train - 96 - ------------>Save best model
...
```
- Test
```yaml
2022-06-02 00:44:01,213 - INFO - main.py - <module> - 253 - ======== Carry Out Testing========
2022-06-02 00:45:17,446 - INFO - main.py - <module> - 257 - 【test】 loss：2.787870 accuracy：0.3636 micro_f1：0.3636 macro_f1：0.4325
2022-06-02 00:45:17,775 - INFO - main.py - <module> - 259 -
                        precision    recall  f1-score   support

            Negative       0.00      0.00      0.00         3
    Cause_of_disease       0.40      0.50      0.44         4
Treatment_of_disease       0.25      0.33      0.29         3
         Association       1.00      1.00      1.00         1

            accuracy                           0.36        11
           macro avg       0.41      0.46      0.43        11
        weighted avg       0.30      0.36      0.33        11
...
```
- Predict
```yaml
2022-06-02 00:45:17,853 - INFO - main.py - <module> - 263 - ======== Prediction ========
2022-06-02 00:45:33,249 - INFO - main.py - <module> - 275 - Halothane is known to oppose <e1start> digitalis <e1end>-induced <e2start> ventricular arrhythmias <e2end>. 
2022-06-02 00:45:35,858 - INFO - main.py - <module> - 277 - predict labels：Association
2022-06-02 00:45:35,874 - INFO - main.py - <module> - 278 - true label：Cause_of_disease
2022-06-02 00:45:35,874 - INFO - main.py - <module> - 279 - ==========================
2022-06-02 00:45:35,874 - INFO - main.py - <module> - 275 - Both cases proved to be <e1start> cotton <e1end>-material-induced <e2start> granulomas <e2end>. 
2022-06-02 00:45:38,171 - INFO - main.py - <module> - 277 - predict labels：Cause_of_disease
2022-06-02 00:45:38,171 - INFO - main.py - <module> - 278 - true label：Cause_of_disease
2022-06-02 00:45:38,186 - INFO - main.py - <module> - 279 - ==========================
2022-06-02 00:45:38,186 - INFO - main.py - <module> - 275 - The evidence for <e1start> soybean <e1end> products as <e2start> cancer <e2end> preventive agents.  
2022-06-02 00:45:40,342 - INFO - main.py - <module> - 277 - predict labels：Cause_of_disease
2022-06-02 00:45:40,342 - INFO - main.py - <module> - 278 - true label：Treatment_of_disease
2022-06-02 00:45:40,342 - INFO - main.py - <module> - 279 - ==========================
2022-06-02 00:45:40,342 - INFO - main.py - <module> - 275 - [Mortality trends in <e2start> cancer <e2end> attributable to <e1start> tobacco <e1end> in Mexico].  
2022-06-02 00:45:42,749 - INFO - main.py - <module> - 277 - predict labels：Association
2022-06-02 00:45:42,749 - INFO - main.py - <module> - 278 - true label：Cause_of_disease
2022-06-02 00:45:42,749 - INFO - main.py - <module> - 279 - ==========================
2022-06-02 00:45:42,749 - INFO - main.py - <module> - 275 - <e1start> Areca <e1end> nut chewing has a significant association with <e2start> systemic inflammation <e2end>.
2022-06-02 00:45:45,038 - INFO - main.py - <module> - 277 - predict labels：Cause_of_disease
2022-06-02 00:45:45,038 - INFO - main.py - <module> - 278 - true label：Association
2022-06-02 00:45:45,038 - INFO - main.py - <module> - 279 - ==========================
2022-06-02 00:45:45,054 - INFO - main.py - <module> - 275 - <e2start> major depression <e2end> (MD) and regular <e1start> tobacco <e1end> use (RU) or nicotine dependence (ND).
2022-06-02 00:45:47,225 - INFO - main.py - <module> - 277 - predict labels：Association
2022-06-02 00:45:47,225 - INFO - main.py - <module> - 278 - true label：Association
2022-06-02 00:45:47,225 - INFO - main.py - <module> - 279 - ==========================
```

# Model output
[view model](https://drive.google.com/drive/folders/1puazcgbXzkk0o4zUPjbFhUSHsHWL4jhT?usp=sharing)

`Best.pt`
model created by pytorch
