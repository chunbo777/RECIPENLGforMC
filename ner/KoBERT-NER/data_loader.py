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
            # raw_docs = re.split(r"\n\t?\n", raw_text)
            raw_docs = re.split(r"[\n][#]{2}[\w]+[\n]", raw_text)
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
                    # 자동 태깅 과정에서 오류 발생 말린 밤(황률) => 말린  밤 황률)
                    # ex) '()' 표시가 없어짐 말린 밤(황률) => 말린  밤 황률)
                    continue 
                token, tag = line.split("\t")
                tags.append(self.labels_lst.index(tag) if tag in self.labels_lst else self.labels_lst.index("UNK"))
                chars.append(token)

            assert len(chars) == len(tags)

            # KoBERTNER
            # words, labels = data.split('\t')# 문장과 label이 한 열에 저장
            # words = words.split()# 각문장과 label을 공백을 기준으로 구분
            # labels = labels.split()# len(words) == len(labels) ??
            # guid = "%s-%s" % (set_type, i)

            # labels_idx = []
            # for label in labels:
            #     # label을 index로 전환
            #     labels_idx.append(self.labels_lst.index(label) if label in self.labels_lst else self.labels_lst.index("UNK"))

            # assert len(words) == len(labels_idx)# ?? 이게 보장이 되려면 공백기준으로 문장을 나누어 나오는 단어 목록의 길이와 label목록의 길이가 동일해야 함

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
                                 sequence_a_segment_id=0,#일반토큰 + sep token동시 고려
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
        # 공백은 tagging이 되어 있어도 고려 안함
        original_clean_labels =  [example.labels[i] for i, char in enumerate(example.words) if char !=' ']  # clean labels (bert clean func)
        sentence = ''.join(example.words)# sentence에 모은 상태에서 split과 tokenizer를 이용해 token 재생성
        # sentence: "안녕 하세요.."
        # original_clean_labels: [안, 녕, 하, 세, 요, ., .]
        # sent_words: [안녕, 하세요..]
        sent_words = sentence.split(" ")#len(''.join(sent_words)) == len(original_clean_labels)
        modi_labels = []
        tokens = []
        char_idx = 0
        for word in sent_words:
            # 안녕, 하세요
            correct_syllable_num = len(word)
            tokenized_word = tokenizer.tokenize(word)
            # case1: 음절 tokenizer --> [안, ##녕]
            # case2: wp tokenizer --> [안녕]
            # case3: 음절, wp tokenizer에서 unk --> [unk]
            # unk규칙 --> 어절이 통채로 unk로 변환, 단, 기호는 분리
            contain_unk = True if tokenizer.unk_token in tokenized_word else False
            for i, token in enumerate(tokenized_word):
                token_repl = token.replace('##', "")# 원래 문장에서 추출한 문자만 남음(unk 제외)
                if not token_repl:
                    continue

                if token in ['<','양념']:
                    print(token)
                # 두개 이상의 문자가 하나의 token으로 합쳐진경우 첫 문자의 label을 새로이 생성된 token의 label로 배정
                tokens.append(token)# '##'표시가 없으면 나중에 token을 id로 converting 할때 오류 ex) UNK token이 발생
                modi_labels.append(original_clean_labels[char_idx])
                # token중 일부가 Unknown token일 경우 (진짜?ㅋㅋㅠㅠ=> 진짜,?,ㅋㅋ,ㅠㅠ=>진짜,?,[UNK],[UNK])
                # 첫번째 문자("당")의 tag 정보가 공백 구분자를 통해 생성된 각 토큰의 tag 정보가 됨
                if not contain_unk: 
                    char_idx += len(token_repl)
            if contain_unk:# ㅋㅋ, ㅠㅠ 같은 것들은 unk로 저장
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
        #     # tokenize를 통해 새롭게 생성된 token에 상응하는 만큼 pad_label이 추가? padding이 아니라 앞선 token이 B or I 일경우 연속값은 I이고  O일경우 O로 배정되어야 하는거 아닌가?
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
        token_type_ids = [sequence_a_segment_id] * len(tokens)#? 일반 token과 special 토큰 모두 sequence_a_segment_id로 처리 

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
    # KLUE에서는 pad_token에 대한 label(tag)을 'O'로 주고 있음
    # pad_token_label_id = 1 # O의 label 인덱스

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
