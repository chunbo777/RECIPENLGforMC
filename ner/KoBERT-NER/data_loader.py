import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

from utils import get_labels

# 20210927
import re

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NaverNerProcessor(object):
    """Processor for the Naver NER data set """

    def __init__(self, args):
        self.args = args
        self.labels_lst = get_labels(args)

    @classmethod
    def _read_file(cls, input_file):
        """Read tsv file, and return words and label as list"""
        with open(input_file, "r", encoding="utf-8") as f:

            # lines = []
            # for line in f:
            #     lines.append(line.strip())
            # return lines

            raw_text = f.read().strip()

            ## 20211001
            raw_docs = re.split(r"[\n][#]{2}[\w]+[\n]", raw_text)# recipe
            # raw_docs = re.split(r"\n\t?\n", raw_text)# klue
            return raw_docs

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            # 20210928
            tags = []  # clean labels (bert clean func)
            chars = []
            for line in data.split("\n"):
                if line.startswith("##"):  # skip comment
                    continue
                elif len(line.split("\t")) !=2:
                    # ?????? ?????? ???????????? ?????? ?????? ?????? ???(??????) => ??????  ??? ??????)
                    # ex) '()' ????????? ????????? ?????? ???(??????) => ??????  ??? ??????)
                    continue 
                token, tag = line.split("\t")
                tags.append(self.labels_lst.index(tag) if tag in self.labels_lst else self.labels_lst.index("UNK"))
                chars.append(token)

            assert len(chars) == len(tags)

            # KoBERTNER
            # words, labels = data.split('\t')# ????????? label??? ??? ?????? ??????
            # words = words.split()# ???????????? label??? ????????? ???????????? ??????
            # labels = labels.split()# len(words) == len(labels) ??
            # guid = "%s-%s" % (set_type, i)

            # labels_idx = []
            # for label in labels:
            #     # label??? index??? ??????
            #     labels_idx.append(self.labels_lst.index(label) if label in self.labels_lst else self.labels_lst.index("UNK"))

            # assert len(words) == len(labels_idx)# ?? ?????? ????????? ????????? ?????????????????? ????????? ????????? ????????? ?????? ????????? ????????? label????????? ????????? ???????????? ???

            if i % 10000 == 0:
                logger.info(data)

            # examples.append(InputExample(guid=guid, words=words, labels=labels_idx))
            examples.append(InputExample(guid=guid, words=chars, labels=tags))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


processors = {
    "naver-ner": NaverNerProcessor,
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,#???????????? + sep token?????? ??????
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # 20210928
        # ????????? tagging??? ?????? ????????? ?????? ??????
        original_clean_labels =  [example.labels[i] for i, char in enumerate(example.words) if char.strip() !='']  # clean labels (bert clean func)
        sentence = ''.join(example.words)# sentence??? ?????? ???????????? split??? tokenizer??? ????????? token ?????????
        # sentence: "?????? ?????????.."
        # original_clean_labels: [???, ???, ???, ???, ???, ., .]
        # sent_words: [??????, ?????????..]
        sent_words = sentence.split(" ")#len(''.join(sent_words)) == len(original_clean_labels)
        modi_labels = []
        tokens = []
        char_idx = 0
        for word in sent_words:
            # ??????, ?????????
            correct_syllable_num = len(word)
            tokenized_word = tokenizer.tokenize(word)
            # case1: ?????? tokenizer --> [???, ##???]
            # case2: wp tokenizer --> [??????]
            # case3: ??????, wp tokenizer?????? unk --> [unk]
            # unk?????? --> ????????? ????????? unk??? ??????, ???, ????????? ??????
            contain_unk = True if tokenizer.unk_token in tokenized_word else False
            for i, token in enumerate(tokenized_word):
                token_repl = token.replace('##', "")# ?????? ???????????? ????????? ????????? ??????(unk ??????)
                if not token_repl:
                    continue

                # if token in ['<','??????']:
                #     print(token)
                # ?????? ????????? ????????? ????????? token?????? ??????????????? ??? ????????? label??? ????????? ????????? token??? label??? ??????
                tokens.append(token)# '##'????????? ????????? ????????? token??? id??? converting ?????? ?????? ex) UNK token??? ??????
                modi_labels.append(original_clean_labels[char_idx])
                # token??? ????????? Unknown token??? ?????? (???????????????????=> ??????,?,??????,??????=>??????,?,[UNK],[UNK])
                # ????????? ??????("???")??? tag ????????? ?????? ???????????? ?????? ????????? ??? ????????? tag ????????? ???
                if not contain_unk: 
                    char_idx += len(token_repl)
            if contain_unk:# ??????, ?????? ?????? ????????? unk??? ??????
                char_idx += correct_syllable_num# = len(word)
        
        assert len(tokens) == len(modi_labels)
        # # Tokenize word by word (for NER)
        # tokens = []
        # label_ids = []
        # for word, slot_label in zip(example.words, example.labels):
        #     word_tokens = tokenizer.tokenize(word)
        #     if not word_tokens:
        #         word_tokens = [unk_token]  # For handling the bad-encoded word
        #     tokens.extend(word_tokens)
        #     # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        #     # tokenize??? ?????? ????????? ????????? token??? ???????????? ?????? pad_label??? ??????? padding??? ????????? ?????? token??? B or I ????????? ???????????? I??????  O????????? O??? ??????????????? ????????? ??????????
        #     label_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))


        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            # label_ids = label_ids[: (max_seq_len - special_tokens_count)]
            modi_labels = modi_labels[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        # label_ids += [pad_token_label_id]
        modi_labels += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)#? ?????? token??? special ?????? ?????? sequence_a_segment_id??? ?????? 

        # Add [CLS] token
        tokens = [cls_token] + tokens
        # label_ids = [pad_token_label_id] + label_ids
        modi_labels = [pad_token_label_id] + modi_labels
        token_type_ids = [cls_token_segment_id] + token_type_ids#?

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        # label_ids = label_ids + ([pad_token_label_id] * padding_length)
        modi_labels = modi_labels + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        # assert len(label_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(label_ids), max_seq_len)
        assert len(modi_labels) == max_seq_len, "Error with slot labels length {} vs {}".format(len(modi_labels), max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            # logger.info("label: %s " % " ".join([str(x) for x in label_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in modi_labels]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                        #   label_ids=label_ids
                          label_ids=modi_labels
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len, mode)

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    # KLUE????????? pad_token??? ?????? label(tag)??? 'O'??? ?????? ??????
    # pad_token_label_id = 1 # O??? label ?????????

    cached_features_file = os.path.join(args.data_dir, cached_file_name)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    assert all_input_ids.count_nonzero() == all_attention_mask.count_nonzero(), "Error with attention_mask"
    assert all_input_ids.count_nonzero() == all_attention_mask.sum(), "Error with attention_mask"
    assert all_token_type_ids.count_nonzero() == 0, "Error with token_type_ids"



    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    return dataset
