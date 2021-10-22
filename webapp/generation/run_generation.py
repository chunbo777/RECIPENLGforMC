#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with GPT-2
"""

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config

from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizerFast

import os

import json

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'kogpt2': (GPT2LMHeadModel, PreTrainedTokenizerFast),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0# False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]# softmax 결과값이 0에 가까운 token 제외
        logits[indices_to_remove] = filter_value# 불필요하다 생각되는 token의 값에 음의 무한대 값 배정
    return logits


def sample_sequence(model, length, context, tokenizer, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    end_token = tokenizer.convert_tokens_to_ids(["<RECIPE_END>"])[0]
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)#torch.Size([1, 8, 50270])  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature# 마지막 tensor 값, torch.Size([50270])
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)# 중요도가 떨어지는 반환값은 음의 무한대로 수정됨
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == end_token or (generated.shape[1] == 1024):
                break
    return generated

import re
from datetime import datetime

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def main(ingredients=None):

    args = AttrDict()
    args.update({
        'model_type':'gpt2', 'model_name_or_path':'mbien/recipenlg'
        # 'model_type':'kogpt2', 'model_name_or_path':f'{os.path.dirname(__file__)}/model'
        , 'prompt':'', 'length':2048, 'temperature':1.0, 'top_k':0, 'top_p':0.9, 'no_cuda':'','seed' : 42 
    })

    print(args)
    start = datetime.now()
    print('start : ', start)


    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    while True:
        if ingredients is None:
            raw_text = args.prompt if args.prompt else input("Comma-separated ingredients, semicolon to close the list >>> ")
        else:
            raw_text = ingredients
        prepared_input = '<RECIPE_START> <INPUT_START> ' + raw_text.replace(',', ' <NEXT_INPUT> ').replace(';', ' <INPUT_END>')
        context_tokens = tokenizer.encode(prepared_input)
        out = sample_sequence(
            model=model,
            context=context_tokens,
            tokenizer=tokenizer,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        out = out[0, len(context_tokens):].tolist()
        text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
        text = text.replace('\\u00b0','\u00b0').replace('\\u00bc','\u00bc')
        if "<RECIPE_END>" not in text:
            print(text)
            print("Failed to generate, recipe's too long")
            continue
        full_text = prepared_input + text
        
        jsonData = {'TITLE':None, 'INPUT':None,'INGR':None,'INSTR':None}
        for k, v in jsonData.items():
             tmp = re.split(f'<NEXT_{k}>',re.sub(f'<{k}_START>|<{k}_END>','',re.search(f"<{k}_START>.*<{k}_END>", full_text).group(0)))
             jsonData.update({k:[i.strip() for i in tmp] if len(tmp)>1 else tmp[0]})
        if args.prompt or ingredients is not None:
            break

    end = datetime.now()
    print('end : ', end)
    print('time : ', start - end)
    return jsonData


if __name__ == '__main__':
    # main()
    # text = main('milk')
    text = main()
    print(text)
