# Draw a bar chart of frequency statistics of plant-disease relationships
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

# Read Excel data
df = pd.read_excel("gold-standard-corpus.xlsx")
label_list = list(df['relation'].value_counts().index)
num_list = df['relation'].value_counts().tolist()

plt.rcParams['font.family'] = ['DejaVu Sans']

# Drawing bar charts with Matplotlib
x = range(len(num_list))
rects = plt.bar(x=x, height=num_list, width=0.6, color='blue', label="Frequency")
plt.ylim(0, 800)  # y-axis range
plt.ylabel("Quantity")
plt.xticks([index + 0.1 for index in x], label_list)
plt.xticks(rotation=10)  # x-axis labels rotated 45 degrees
plt.xlabel("Entities Relationships")
plt.title("Plant Disease Relationship Frequency Statistics")
plt.legend()

# Text description
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")

# Show Chart
# plt.show()

# Save Chart
plt.savefig('bar_chart.png')
