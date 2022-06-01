import re
import json
import pandas as pd
from pprint import pprint

df = pd.read_excel('gold-standard-corpus.xlsx')
relations = list(df['relation'].unique())

relations.remove('Negative')
relation_dict = {'Negative': 0}
relation_dict.update(dict(zip(relations, range(1, len(relations) + 1))))

with open('rel_dict.json', 'w', encoding='utf-8') as h:
    h.write(json.dumps(relation_dict, ensure_ascii=False, indent=2))

pprint(df['relation'].value_counts())

print("============================")
print('total data : %s' % len(df))
print("\n")

df['rel'] = df['relation'].apply(lambda x: relation_dict[x])

texts = []

print(" Example Data")
print("""
id_relation, <e1>entity1</e1>, <e2>entity2</e2>, sentence, start_entity1, end_entity1, start_entity2, end_entity2\n\n
""")

for per1, per2, text, label, e1start, e1end, e2start, e2end in zip(
        df['plant'].tolist(),
        df['disease'].tolist(),
        df['sentence'],
        df['rel'].tolist(),
        df['e1start'].tolist(),
        df['e1end'].tolist(),
        df['e2start'].tolist(),
        df['e2end'].tolist()
):
    text = f"{text}\t{e1start}\t{e1end}\t{e2start}\t{e2end}"
    texts.append([text, label])

df = pd.DataFrame(texts, columns=['text', 'rel'])
df['length'] = df['text'].apply(lambda x: len(x))
df = df[df['length'] <= 128]

predict_df = df.sample(frac=0.8, random_state=1024)
train_df = df.sample(frac=0.8, random_state=1024)
test_df = df.drop(train_df.index)

with open('predict.txt', 'w', encoding='utf-8') as f:
    for text, rel in zip(predict_df['text'].tolist(), predict_df['rel'].tolist()):
        f.write(str(rel) + '\t' + text + '\n')

with open('train.txt', 'w', encoding='utf-8') as f:
    for text, rel in zip(train_df['text'].tolist(), train_df['rel'].tolist()):
        f.write(str(rel) + '\t' + text + '\n')

with open('test.txt', 'w', encoding='utf-8') as g:
    for text, rel in zip(test_df['text'].tolist(), test_df['rel'].tolist()):
        g.write(str(rel) + '\t' + text + '\n')
