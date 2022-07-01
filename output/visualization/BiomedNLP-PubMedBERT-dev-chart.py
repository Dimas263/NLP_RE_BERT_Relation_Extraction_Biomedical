import matplotlib.pyplot as plt
import pandas as pd
import csv

dev_url = 'drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/visualization/BiomedNLP-PubMedBERT-dev.csv'

biobert_dev = pd.read_csv(dev_url, header=0)

x = biobert_dev['Loss']
y = biobert_dev['macro_f1']

plt.plot(y, color='g', linestyle='dashed', marker='o', label="Dev F-1 Scores")

plt.xticks(rotation=25)
plt.xlabel('Steps')
plt.ylabel('F-1')
plt.title('Development', fontsize=20)
plt.grid()
plt.legend()
# plt.show()
plt.savefig('drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/visualization/BiomedNLP-PubMedBERT-dev.png')
