from pprint import pprint
import os
import logging
import json
import shutil

from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, AutoTokenizer

import config
import preprocess_no_log
import dataset
import models
import utils

logger = logging.getLogger(__name__)

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()


class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.model = models.BertForRelationExtraction(args)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model.to(self.device)


    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss


    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)


    def train(self):
        total_step = len(self.train_loader) * self.args.train_epochs
        global_step = 0
        eval_step = 1
        best_dev_micro_f1 = 0.0
        for epoch in range(args.train_epochs):
            for train_step, train_data in enumerate(self.train_loader):
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)
                ids = train_data['ids'].to(self.device)
                train_outputs = self.model(token_ids, attention_masks, token_type_ids, ids)
                loss = self.criterion(train_outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                logger.info(
                    "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                global_step += 1
                if global_step % eval_step == 0:
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    accuracy, micro_f1, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                    logger.info(
                        "【dev】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(dev_loss, accuracy, micro_f1, macro_f1))
                    if macro_f1 > best_dev_micro_f1:
                        logger.info("------------>Save best model")
                        checkpoint = {
                            'epoch': epoch,
                            'loss': dev_loss,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        best_dev_micro_f1 = macro_f1
                        checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
                        self.save_ckp(checkpoint, checkpoint_path)
                

    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)
                ids = dev_data['ids'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids, ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        model = self.model
        optimizer = self.optimizer
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, checkpoint_path)
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                ids = test_data['ids'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids, ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, id2label, args, ids):
        model = self.model
        optimizer = self.optimizer
        checkpoint = os.path.join(args.output_dir, 'best.pt')
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, checkpoint)
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            inputs = tokenizer.encode_plus(
                text=text,
                add_special_tokens=True,
                max_length=args.max_seq_len,
                truncation='longest_first',
                padding="max_length",
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            token_ids = inputs['input_ids'].to(self.device).long()
            attention_masks = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)
            ids = torch.from_numpy(np.array([[x + 1 for x in ids]])).to(self.device)
            outputs = model(token_ids, attention_masks, token_type_ids, ids)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten().tolist()
            if len(outputs) != 0:
                outputs = [id2label[i] for i in outputs]
                return outputs
            else:
                return 'sorry, i didnt recognize it'

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, micro_f1, macro_f1

    def get_classification_report(self, outputs, targets, labels):
        report = classification_report(targets, outputs, target_names=labels)
        return report


if __name__ == '__main__':
    args = config.Args().get_parser()
    utils.utils.set_seed(args.seed)
    utils.utils.set_logger(os.path.join(args.main_log_dir))

    processor = preprocess_no_log.Processor()

    label2id = {}
    id2label = {}
    with open('drive/MyDrive/Colab Notebooks/bert_relation_extraction/input/data/rel_dict.json', 'r') as fp:
        labels = json.loads(fp.read())
    for k, v in labels.items():
        label2id[k] = v
        id2label[v] = k
    logger.info(label2id)

    logger.info('======== Prediction ========')
    trainer = Trainer(args, None, None, None)
    checkpoint_path = 'drive/MyDrive/Colab Notebooks/bert_relation_extraction/output/checkpoint/best.pt'
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir, local_files_only=True, ignore_mismatched_sizes=True)
    # tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, local_files_only=True, ignore_mismatched_sizes=True)

    with open(os.path.join('drive/MyDrive/Colab Notebooks/bert_relation_extraction/input/data/predict.txt'), 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split('\t')
            label = line[0]
            text = line[1]
            ids = [int(line[2]), int(line[3]), int(line[4]), int(line[5])]
            logger.info(text)
            result = trainer.predict(tokenizer, text, id2label, args, ids)
            logger.info("predict labels：" + "".join(result))
            logger.info("true label：" + id2label[int(line[0])])
            logger.info("==========================")

    # text = '1	Halothane is known to oppose <e1start> digitalis <e1end>-induced <e2start> ventricular arrhythmias <e2end>. 	29	56	65	106'
    text = 'Halothane is known to oppose <e1start> digitalis <e1end>-induced <e2start> ventricular arrhythmias <e2end>.'
    ids = [29, 56, 65, 106]
    print(text)
    print('predict labels：', trainer.predict(tokenizer, text, id2label, args, ids))
    print('true label：', id2label[1])

