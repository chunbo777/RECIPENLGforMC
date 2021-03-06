import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from tqdm import tqdm

from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    ElectraTokenizer,
    BertTokenizer,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification
)
from tokenization_kobert import KoBertTokenizer

import re


MODEL_CLASSES = {
    'kobert': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertForTokenClassification, KoBertTokenizer),
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'koelectra-base': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
    'koelectra-small': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),

    'koelectra-base_v3': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
}

MODEL_PATH_MAP = {
    'kobert': 'monologg/kobert',
    'distilkobert': 'monologg/distilkobert',
    'bert': 'bert-base-multilingual-cased',
    'kobert-lm': 'monologg/kobert-lm',
    'koelectra-base': 'monologg/koelectra-base-discriminator',
    'koelectra-small': 'monologg/koelectra-small-discriminator',

    'koelectra-base_v3': 'monologg/koelectra-base-v3-discriminator',
}


def get_test_texts(args, tokenizer):
    texts = []
    with open(os.path.join(args.data_dir, args.test_file), 'r', encoding='utf-8') as f:
        # for line in f:
        #     text, _ = line.split('\t')
        #     text = text.split()
        #     texts.append(text)
        raw_text = f.read().strip()

        # 20211001
        raw_docs = re.split(r"[\n][#]{2}[\w]+[\n]", raw_text)# 실제 사용하는 data
        # raw_docs = re.split(r"\n\t?\n", raw_text)# klue

        for data in tqdm(raw_docs):
            chars = []
            for line in data.split("\n"):
                if line.startswith("##"):  # skip comment
                    continue
                elif len(line.split("\t")) !=2:
                # 자동 태깅 과정에서 오류 발생 
                # ex) '()' 표시가 없어짐 말린 밤(황률) => 말린  밤 황률)
                # 이거 대신 원문을 사용해보기
                    continue 
                token, _ = line.split("\t")
                chars.append(token)
            sent_words = ''.join(chars).split(" ")
            tokens = []
            for word in sent_words:
                tokenized_word = tokenizer.tokenize(word)
                for i, token in enumerate(tokenized_word):
                    token_repl = token.replace('##', "")# 원래 문장에서 추출한 문자만 남음(unk 제외)
                    if not token_repl:
                        continue
                    tokens.append(token)# '##'표시가 없으면 나중에 token을 id로 converting 할때 오류 ex) UNK token이 발생
            texts.append(tokens)
    return texts


def get_labels(args):

    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]

    # # klue 학습 과정 확인중
    # return [
    #     'UNK', 'O',#20211001
    #     "B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG",
    #     "B-DT", "I-DT", "B-TI", "I-TI", "B-QT", "I-QT",
    #     # "O",
    # ]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        # level=logging.INFO)
                        level=logging.DEBUG)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec(labels, preds)


def f1_pre_rec(labels, preds):
    # return {
    #     "precision": precision_score(labels, preds, suffix=True),
    #     "recall": recall_score(labels, preds, suffix=True),
    #     "f1": f1_score(labels, preds, suffix=True)
    # }
    return {
        "precision": precision_score(labels, preds, suffix=False),
        "recall": recall_score(labels, preds, suffix=False),
        "f1": f1_score(labels, preds, suffix=False)
    }


def show_report(labels, preds):
    # return classification_report(labels, preds, suffix=True)
    return classification_report(labels, preds, suffix=False)
