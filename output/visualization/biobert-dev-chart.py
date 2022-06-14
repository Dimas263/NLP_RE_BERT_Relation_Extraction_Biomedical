import matplotlib.pyplot as plt
import csv

x = []
y = []

url = 'drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/visualization/biobert-dev.csv'

with open(url, 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        x.append(float(row[4]))
        y.append(float(row[2]))

plt.plot(x, y, color='g', linestyle='None',
         marker='o', label="Dev Accuracy")

plt.xticks(rotation=25)
plt.xlabel('F-1')
plt.ylabel('Accuracy')
plt.title('Development', fontsize=20)
plt.grid()
plt.legend()
# plt.show()
plt.savefig('drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/visualization/biobert-dev.png')
