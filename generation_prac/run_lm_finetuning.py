# -*- coding: utf-8 -*-
#20210827
"""
Fine-tuning the library models for language modeling on a text file (GPT-2,).
GPT-2 is fine-tuned using a causal language modeling (CLM) loss
"""

import argparse
import glob
import logging
import os
import random
import gc
import h5py
import boto3
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)

logger = logging.getLogger(__name__)


import wandb
wandb.login()

import tarfile
def tardir(path, tar_name):
    with tarfile.open(tar_name, "w") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        # cached_features_file = "unsupervised.h5"
        # cached_features_file = "/home/lab17/recipe_generation/recipenlg/generation/datain/unsupervised_short.h5"
        # cached_features_file = "/home/lab17/recipe_generation/recipenlg/generation/datain/unsupervised.h5"
        # cached_features_file = "/home/lab17/RECIPENLGforMC/generation_prac/datain/unsupervised_translated.h5"
        cached_features_file = "/home/lab17/RECIPENLGforMC/generation_prac/datain/unsupervised_translated_short.h5"

        logger.info("Loading features from cached file %s", cached_features_file)
        with h5py.File(cached_features_file, 'r') as f:
            if file_path=='test':
                # self.examples = f[file_path][:] #this is a dev set, 10% of a test set
                # self.examples = f['unsupervised_test_translated'][:] # 저장할때 사용한 파일명
                self.examples = f['unsupervised_test_kr_1m_translated_short'][:] # 저장할때 사용한 파일명
            else:
                # self.examples = f[file_path][:]
                # self.examples = f['unsupervised_train_translated'][:]#  저장할때 사용한 파일명
                self.examples = f['unsupervised_train_kr_1m_translated_short'][:]#  저장할때 사용한 파일명

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # return torch.tensor(self.examples[item])
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):

    dataset = TextDataset(tokenizer, file_path="test" if evaluate else "train", block_size= args.block_size)
    return dataset


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    '''
    AMP automatically refactors PyTorch computational graphs by replacing some of FP32 tensors with FP16 ones. 
        This operation aims to both decrease model memory usage and training times
        , while it could negatively affect model precision and loss function decreases. 
        Apex AMP offers four levels of optimization (O0 - O3). 
        O2 was selected and used for all the training operations on GPUs.
    '''

    try:
        from apex import amp# 여기서는 gpu가 없어 안돌아감
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")#>>1278MiB

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            # logger.info(step)# 진행상황에 찍힘
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()

            outputs = model(inputs, labels=labels)#7389MiB
            # loss = outputs[0] # model outputs are always tuple in transformers (see doc)
            loss = outputs['loss'] #20210830

            wandb.log({f"loss": loss})#20210912
            

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            tr_loss += loss.item()
            if (step) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                wandb.log({f"learning_rate": scheduler.get_lr()[0]})#20210912

                model.zero_grad()
                global_step += 1
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            wandb.log({f"evaluate_during_training_{key}": value})#20210912
                    if args.aws_bucket:
                        tgz = "checkpoint-{}.tar".format(global_step)
                        tardir(output_dir, tgz)
                        shutil.rmtree(output_dir)
                        s3 = boto3.resource('s3')
                        s3.Object(args.aws_bucket, "checkpoints-gpt-medium/"+tgz).upload_file(tgz)
                        os.remove(tgz)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = batch.to(args.device)

        with torch.no_grad():
            outputs = model(batch, labels=batch)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    '''
    20210902
    Perplexity is a measure that shows how well a probability distribution can predict a sample.
    the measure is commonly used to show model fitness to particular task. 
    However, it is known that the model with lower perplexity level could sometimes be 
    less convincing for humans that the one that generalizes worse (and has higher perplexity). 
    As such, perplexity is a good automatic metric 
    that cannot be treated as a single source of truth about model performance.
    '''
    perplexity = torch.exp(torch.tensor(eval_loss))
    wandb.log({'val_loss': eval_loss, 'perplexity': perplexity}) # 20210913

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():

    #Run pjgba7co errored: ValueError('You must call `wandb.init` before calling watch')
    wandb.init(project='with translated') 

    parser = argparse.ArgumentParser()
    ## Required parameters
    # parser.add_argument("--train_data_file", default=None, type=str, required=True,
    #                     help="The input training data file (a text file).")
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    #20210826 default 수정 및 required property 미사용 처리
    # parser.add_argument("--train_data_file", default='/home/lab17/recipe_generation/recipenlg/generation/datain/unsupervised_short.h5', type=str, help="The input training data file (a text file).")
    # parser.add_argument("--train_data_file", default='/home/lab17/recipe_generation/recipenlg/generation/datain/unsupervised.h5', type=str, help="The input training data file (a text file).")
    # parser.add_argument("--train_data_file", default='/home/lab17/RECIPENLGforMC/generation_prac/datain/unsupervised_translated.h5', type=str, help="The input training data file (a text file).")
    parser.add_argument("--train_data_file", default='/home/lab17/RECIPENLGforMC/generation_prac/datain/unsupervised_translated_short.h5', type=str, help="The input training data file (a text file).")
    # parser.add_argument("--output_dir", default='/home/lab17/recipe_generation/recipenlg/output_short/', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir", default='/home/lab17/RECIPENLGforMC/generation_prac/dataout/', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--output_dir", default='/home/lab17/recipe_generation/recipenlg/output_gpt2/', type=str, help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    # parser.add_argument("--eval_data_file", default=None, type=str,
    #                     help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--eval_data_file", default='/home/lab17/RECIPENLGforMC/generation_prac/dataout/eval_data_file.txt', type=str)

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")

    # 20210826
    # parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
    #                     help="The model checkpoint for weights initialization.")
    parser.add_argument("--model_name_or_path", default="gpt2", type=str, help="The model checkpoint for weights initialization.")
    # parser.add_argument("--model_name_or_path", default="mbien/recipenlg", type=str, help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    # 20210827 git 참고
    # parser.add_argument("--tokenizer_name", default="", type=str,
    #                     help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="gpt2", type=str, help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    # parser.add_argument("--tokenizer_name", default="mbien/recipenlg", type=str, help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")


    # parser.add_argument("--block_size", default=-1, type=int,
    #                     help="Optional input sequence length after tokenization."
    #                          "The training dataset will be truncated in block of this size for training."
    #                          "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--block_size", default=wandb.config.block_size if 'block_size' in wandb.config.keys() and isinstance(wandb.config.block_size, int) else -1, type=int)


    # parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', default=True, help="Whether to run training.")#20210826

    # parser.add_argument("--do_eval", action='store_true',
    #                     help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval", action='store_true', default=True)

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    #20210831
    # parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
    #                     help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_train_batch_size", type=int
    , default=wandb.config.per_gpu_train_batch_size if 'per_gpu_train_batch_size' in wandb.config.keys() and isinstance(wandb.config.per_gpu_train_batch_size, int) else 4
    , help="Batch size per GPU/CPU for training.")

    # parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
    #                     help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--per_gpu_eval_batch_size", type=int
    , default=wandb.config.per_gpu_eval_batch_size if 'per_gpu_eval_batch_size' in wandb.config.keys() and isinstance(wandb.config.per_gpu_eval_batch_size, int) else 4
    , help="Batch size per GPU/CPU for evaluation.")


    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--gradient_accumulation_steps', type=int
    , default=wandb.config.gradient_accumulation_steps if 'gradient_accumulation_steps' in wandb.config.keys() and isinstance(wandb.config.gradient_accumulation_steps, int) else 1)

    # parser.add_argument("--learning_rate", default=5e-5, type=float,
    #                     help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate", type=float
    , default=wandb.config.learning_rate if 'learning_rate' in wandb.config.keys() and isinstance(wandb.config.learning_rate, float) else 5e-5)

    # parser.add_argument("--weight_decay", default=0.0, type=float,
    #                     help="Weight deay if we apply some.")
    parser.add_argument("--weight_decay", type=float
    , default=wandb.config.weight_decay if 'weight_decay' in wandb.config.keys() and isinstance(wandb.config.weight_decay, float) else 0.0)


    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float
    , default=wandb.config.adam_epsilon if 'adam_epsilon' in wandb.config.keys() and isinstance(wandb.config.adam_epsilon, float) else 1e-8)

    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    parser.add_argument("--max_grad_norm", type=float
    , default=wandb.config.max_grad_norm if 'max_grad_norm' in wandb.config.keys() and isinstance(wandb.config.max_grad_norm, float) else 1.0)
    
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    # parser.add_argument("--num_train_epochs", default=2.0, type=float, help="Total number of training epochs to perform.")

    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    # parser.add_argument("--max_steps", default=5, type=int)

    
    # parser.add_argument("--warmup_steps", default=0, type=int,
    #                     help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_steps", type=int
    , default=wandb.config.warmup_steps if 'warmup_steps' in wandb.config.keys() and isinstance(wandb.config.warmup_steps, int) else 0)

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    # 20210901
    # parser.add_argument('--save_steps', type=int, default=50,
    #                     help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2000, help="Save checkpoint every X updates steps.")

    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    # 20210831
    # parser.add_argument('--overwrite_output_dir', action='store_true',
    #                     help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_output_dir', action='store_true', default=True, help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--aws_bucket", default="", type=str,
                        help="Whether to upload to specified bucket.")
    args = parser.parse_args()


    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    # logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                     datefmt = '%m/%d/%Y %H:%M:%S',
    #                     level = logging.INFO)
    # 20210826
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.DEBUG)

    '''
    20210902
    it was decided that the available computational capacity should be used 
        to continue training on the small models rather than bidirectional one. 
    '''
    model_class = GPT2LMHeadModel
    model = model_class.from_pretrained(args.model_name_or_path)
    wandb.watch(model)#20210913
    '''
    20210902
    While pretrained GPT-2 offers rich dictionary of language tokens
    , the structure generation task enforced use of additional, custom control tokens. 
    '''
    tokenizer_class = GPT2Tokenizer
    temp = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    # 이게 추가적으로 조정되는것이 아니라면 finedtuned 된 vocab 정보를 가지고 오는게 필요해 보임
    tokenizer = tokenizer_class.from_pretrained(temp, do_lower_case=args.do_lower_case)
    special_tokens = {
        # "additional_special_tokens": [
        #     "<RECIPE_START>", "<RECIPE_END>",
        #     "<INPUT_START>", "<INPUT_END>", "<NEXT_INPUT>"
        #     "<INGR_START>", "<NEXT_INGR>", "<INGR_END>",
        #     "<INSTR_START>", "<NEXT_INSTR>", "<INSTR_END>",
        #     "<TITLE_START>", "<TITLE_END>",
        # ]

        # 20210913 dataset 만들때 순서가 변경된 경우 영향이 있음
        "additional_special_tokens": [
            "<TITLE_START>",
            "<TITLE_END>",
            "<INSTR_START>",
            "<NEXT_INSTR>",
            "<INSTR_END>",
            "<INGR_START>",
            "<NEXT_INGR>",
            "<INGR_END>",
            "<RECIPE_START>",
            "<RECIPE_END>",
            "<INPUT_START>",
            "<INPUT_END>",
            "<NEXT_INPUT>"
        ]
    }

    tokenizer.add_special_tokens(special_tokens)
    '''
    20210902
    Because additional control tokens were added, the embedding layer was also resized?? 
        to learn representation of those during the training process.
    '''
    model.resize_token_embeddings(len(tokenizer))

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model.to(args.device)#1204MiB

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        print(len(train_dataset.examples))
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    # ✍️ Log your final accuracy as a Summary Metric ✍️
    # wandb.run.summary["final_accuracy"] = epoch_acc
    # wandb.run.finish()
    return results

def get_sweep_id(method):
    sweep_config = {
        'method': method
        , 'metric' : {'name':'val_loss', 'goal':'minimize'}
        , 'parameters' : {
            # 'block_size':{
            #     'values':[254,512]
            # },
            'per_gpu_train_batch_size':{
                'values':[3, 4, 5]
            },
            # 'per_gpu_eval_batch_size':{
            #     'values':[3, 4]
            # },
            # 'gradient_accumulation_steps':{
            #     'values':[1,2]
            # },
            # 'learning_rate':{
            #     'values':[1e-5,5e-5,2e-6]
            # },
            # 'weight_decay':{
            #     'values':[0.0,1e-5]
            # },
            # 'adam_epsilon':{
            #     'values':[1e-7,1e-8,1e-7]
            # },
            # 'max_grad_norm':{
            #     'values':[0.5,1.0,1.5]
            # },
            # 'warmup_steps':{
            #     'values':[0,1]
            # },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="cartpole") 
    return sweep_id


if __name__ == "__main__":
    # 20210902
    # main()
    # results = main()
    # 20210912
    sweep_id = get_sweep_id('grid')
    wandb.agent(sweep_id=sweep_id, function=main)
