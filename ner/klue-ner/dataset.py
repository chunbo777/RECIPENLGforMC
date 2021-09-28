from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

class CollateNer(object):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, label2idx: Dict[str, int], max_length: int
    ):
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_length = max_length

    # Dataset subclass의 getItem 함수 실행이후 그 결과(batch)를 넘겨받음
    def __call__(self, input_examples):
        input_texts, input_labels_str = [], []
        for input_example in input_examples:
            text, label_strs = input_example#원문과 tokenizer 정보가 반영된 label 정보
            input_texts.append(text)
            input_labels_str.append(label_strs)

        encoded_texts = self.tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",  # KLUE 베이스라인과 input형태를 일치 시키기 위해
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        #torch.Size([len(input_examples), self.max_length])
        # input_ids.count_nonzero() == attention_mask.sum() == attention_mask.count_nonzero()
        input_ids = encoded_texts["input_ids"]
        attention_mask = encoded_texts["attention_mask"] 

        token_type_ids = encoded_texts["token_type_ids"] # token_type_ids.sum() >> tensor(0)

        len_input = input_ids.size(1)# == self.max_length
        input_labels = []
        for input_label_str in input_labels_str:#len(input_labels_str)은 padding과 special token adding 전 tokens의 길이와 동일
            input_label_str = (#<CLS>, <...>, ...<...>,<SEP>, <PAD>, <PAD>..., <PAD>
                ["O"] + input_label_str + (len_input - len(input_label_str) - 1) * ["O"]
            )# input 형식에 맞춰 Out label 추가(아마 padding token에 대응하는 값일듯)
            input_label = [self.label2idx[x] for x in input_label_str]
            input_label = torch.tensor(input_label).long()
            input_labels.append(input_label)

        input_labels = torch.stack(input_labels)# == torch.Size([len(input_examples), self.max_length])
        return input_ids, token_type_ids, attention_mask, input_labels


class NerDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset: List[Dict[str, Union[str, List[str]]]],
        label_list: List[str],
        max_length: int,
        batch_size: int = None,
        shuffle: bool = False,
        **kwargs
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label2idx = {label: i for i, label in enumerate(label_list)}
        self.collate_fn = CollateNer(tokenizer, self.label2idx, max_length)
        self.loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instance = self.dataset[index]
        text = instance["text_a"]
        label_strs = instance["label"]

        return text, label_strs
