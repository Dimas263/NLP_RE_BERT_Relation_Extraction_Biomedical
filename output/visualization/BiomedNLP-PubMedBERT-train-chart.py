import matplotlib.pyplot as plt
import pandas as pd
import csv

url = 'drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/visualization/BiomedNLP-PubMedBERT-train.csv'

biobert_train = pd.read_csv(url, header=0)

x = biobert_train['Epoch']
y = biobert_train['Loss']

plt.plot(x, y, color='g', linestyle='dashed', marker='o', label="Training Loss", data=None)

plt.xticks(rotation=25)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training', fontsize=20)
plt.grid()
plt.legend()
# plt.show()
plt.savefig('drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/visualization/BiomedNLP-PubMedBERT-train.png')
