import os
import logging
import pickle

from transformers import BertTokenizer
import config
import numpy as np
import json
from utils import utils

logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, set_type, text, labels=None, ids=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels
        self.ids = ids


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class BertFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None, ids=None):
        super(BertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        # labels
        self.labels = labels
        # ids
        self.ids = ids


class Processor:

    @staticmethod
    def read_txt(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = f.read().strip()
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        for line in raw_examples.split('\n'):
            line = line.split('\t')
            if len(line) == 6:
                labels = int(line[0])
                text = line[1]
                ids = [int(line[2]), int(line[3]), int(line[4]), int(line[5])]
                examples.append(
                    InputExample(
                        set_type=set_type,
                        text=text,
                        labels=labels,
                        ids=ids
                    )
                )
        return examples


def convert_bert_example(ex_idx, example: InputExample, tokenizer: BertTokenizer, max_seq_len):
    set_type = example.set_type
    raw_text = example.text
    labels = example.labels
    ids = example.ids
    callback_info = (raw_text,)
    callback_labels = labels
    callback_info += (callback_labels,)

    # label_ids = label2id[labels]
    ids = [x + 1 for x in ids]
    tokens = [i for i in raw_text]
    encode_dict = tokenizer.encode_plus(
        text=tokens,
        add_special_tokens=True,
        max_length=max_seq_len,
        truncation='longest_first',
        padding="max_length",
        return_token_type_ids=True,
        return_attention_mask=True
    )
    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 4:
        decode_text = tokenizer.decode(np.array(token_ids)[np.where(np.array(attention_masks) == 1)[0]].tolist())
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f"text: {decode_text}")
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f"labels: {labels}")
        logger.info(f"ids: {ids}")

    feature = BertFeature(
        # bert inputs
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=labels,
        ids=ids
    )

    return feature, callback_info


def convert_examples_to_features(examples, max_seq_len, bert_dir):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))
    features = []
    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):
        feature, tmp_callback = convert_bert_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
        )
        if feature is None:
            continue

        features.append(feature)
        callback_info.append(tmp_callback)
    logger.info(f'Build {len(features)} features')

    out = (features,)

    if not len(callback_info):
        return out

    out += (callback_info,)
    return out


def get_out(processor, txt_path, args, id2label, mode):
    raw_examples = processor.read_txt(txt_path)

    examples = processor.get_examples(raw_examples, mode)
    for i, example in enumerate(examples):
    
        logger.info("==========================")
        logger.info(f"example_text : {example.text}")
        logger.info(f"example_id_label : {id2label[example.labels]}")
        logger.info(f"example_id_tags : {example.ids}")
        logger.info("==========================")

        if i == 3:
            break
    out = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir)

    def save_pkl(data_dir, data, desc):
        """Save .pkl file"""
        with open(os.path.join(data_dir, '{}.pkl'.format(desc)), 'wb') as f:
            pickle.dump(data, f)

    save_path = os.path.join(args.data_dir, 'final_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_pkl(save_path, out, mode)

    return out


if __name__ == '__main__':
    args = config.Args().get_parser()
    args.log_dir = args.log_dir
    args.max_seq_len = args.max_seq_len 
    args.bert_dir = args.bert_dir
    utils.set_logger(os.path.join(args.preprocess_log_dir))
    logger.info(vars(args))

    processor = Processor()

    label2id = {}
    id2label = {}
    with open('drive/MyDrive/Rearch_Dimas/BERT_RE/input/data/rel_dict.json', 'r') as fp:
        labels = json.loads(fp.read())
    for k, v in labels.items():
        label2id[k] = v
        id2label[v] = v

    train_out = get_out(processor, 'drive/MyDrive/Rearch_Dimas/BERT_RE/input/data/train.txt', args, id2label, 'train')
    dev_out = get_out(processor, 'drive/MyDrive/Rearch_Dimas/BERT_RE/input/data/test.txt', args, id2label, 'dev')
    test_out = get_out(processor, 'drive/MyDrive/Rearch_Dimas/BERT_RE/input/data/test.txt', args, id2label, 'test')
