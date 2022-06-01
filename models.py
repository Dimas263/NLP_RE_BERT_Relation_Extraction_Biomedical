from transformers import BertModel
import torch.nn as nn
import numpy as np
import torch
import config
import json
from torch.utils.data import Dataset, DataLoader, RandomSampler
from dataset import ReDataset
from preprocess import BertFeature
from preprocess import get_out, Processor

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

class BertForRelationExtraction(nn.Module):
    def __init__(self, args):
        super(BertForRelationExtraction, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        self.dropout = nn.Dropout(args.dropout_prob)
        self.linear = nn.Linear(out_dims * 4, args.num_tags)

    def forward(self, token_ids, attention_masks, token_type_ids, ids):
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )
        seq_out = bert_outputs[0]
        batch_size = seq_out.size(0)
        seq_ent = torch.cat(
            [torch.index_select(seq_out[i, :, :], 0, ids[i, :].long()).unsqueeze(0) for i in range(batch_size)], 0)
        print(seq_ent.shape)
        seq_ent = self.dropout(seq_ent)
        seq_ent = seq_ent.view(batch_size, -1)
        seq_ent = self.linear(seq_ent)
        return seq_ent


if __name__ == '__main__':
    args = config.Args().get_parser()
    args.log_dir = 'output/logs/'
    args.max_seq_len = 128
    args.bert_dir = 'model/biobert/'

    processor = Processor()

    label2id = {}
    id2label = {}
    with open('input/data/rel_dict.json', 'r') as fp:
        labels = json.loads(fp.read())
    for k, v in labels.items():
        label2id[k] = v
        id2label[v] = k
    print(label2id)

    train_out = get_out(processor, 'input/data/train.txt', args, id2label, 'train')
    dev_out = get_out(processor, 'input/data/test.txt', args, id2label, 'dev')
    test_out = get_out(processor, 'input/data/test.txt', args, id2label, 'test')

    # import pickle
    # with open('input/data/final_data/train.pkl','wb') as fp:
    #     pickle.dump(train_out, fp)
    # with open('input/data/final_data/dev.pkl','wb') as fp:
    #     pickle.dump(dev_out, fp)
    # with open('input/data/final_data/test.pkl','wb') as fp:
    #     pickle.dump(test_out, fp)
    #
    # train_out = pickle.load(open('input/data/final_data/dev.pkl','rb'))
    train_features, train_callback_info = train_out
    train_dataset = ReDataset(train_features)
    # for data in train_dataset:
    #     print(data['token_ids'])
    #     print(data['attention_masks'])
    #     print(data['token_type_ids'])
    #     print(data['labels'])
    #     print(data['ids'])
    #     break

    args.train_batch_size = 32
    train_dataset = ReDataset(train_features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForRelationExtraction(args)
    model.to(device)
    model.train()
    for step, train_data in enumerate(train_loader):
        # print(train_data['token_ids'].shape)
        # print(train_data['attention_masks'].shape)
        # print(train_data['token_type_ids'].shape)
        # print(train_data['labels'])
        # print(train_data['ids'].shape)
        token_ids = train_data['token_ids'].to(device)
        attention_masks = train_data['attention_masks'].to(device)
        token_type_ids = train_data['token_type_ids'].to(device)
        label_ids = train_data['labels'].to(device)
        ids = train_data['ids'].to(device)
        output = model(token_ids, attention_masks, token_type_ids, ids)
        if step == 1:
            break
