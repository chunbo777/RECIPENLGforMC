import os
import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples
# import wandb
# wandb.login()

def main(args):
# def main(config=None):

    # with wandb.init(config=config):
    #     config = wandb.config

    #     parser = argparse.ArgumentParser()

    #     parser.add_argument("--task", default="naver-ner", type=str, help="The name of the task to train")
    #     parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    #     parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    #     parser.add_argument("--pred_dir", default="./preds", type=str, help="The prediction file dir")

    #     parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    #     parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    #     parser.add_argument("--label_file", default="label.txt", type=str, help="Slot Label file")
    #     parser.add_argument("--write_pred", action="store_true", help="Write prediction during evaluation")

    #     parser.add_argument("--model_type", default=config.model_type if 'model_type' in config.keys() and isinstance(config.model_type, str) else "kobert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    #     parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    #     parser.add_argument("--train_batch_size", default=config.train_batch_size if 'train_batch_size' in config.keys() and isinstance(config.train_batch_size, int) else 32, type=int, help="Batch size for training.")
    #     parser.add_argument("--eval_batch_size", default=config.eval_batch_size if 'eval_batch_size' in config.keys() and isinstance(config.eval_batch_size, int) else 64, type=int, help="Batch size for evaluation.")
    #     parser.add_argument("--max_seq_len", default=config.max_seq_len if 'max_seq_len' in config.keys() and isinstance(config.max_seq_len, int) else 50, type=int, help="The maximum total input sequence length after tokenization.")
    #     parser.add_argument("--learning_rate", default=config.learning_rate if 'learning_rate' in config.keys() and isinstance(config.learning_rate, float) else 5e-5, type=float, help="The initial learning rate for Adam.")
    #     parser.add_argument("--num_train_epochs", default=config.num_train_epochs if 'num_train_epochs' in config.keys() and isinstance(config.num_train_epochs, float) else 20.0, type=float, help="Total number of training epochs to perform.")
    #     parser.add_argument("--weight_decay", default=config.weight_decay if 'weight_decay' in config.keys() and isinstance(config.weight_decay, float) else 0.0, type=float, help="Weight decay if we apply some.")
    #     parser.add_argument('--gradient_accumulation_steps', type=int, default=config.gradient_accumulation_steps if 'gradient_accumulation_steps' in config.keys() and isinstance(config.gradient_accumulation_steps, int) else 1,
    #                         help="Number of updates steps to accumulate before performing a backward/update pass.")
    #     parser.add_argument("--adam_epsilon", default=config.adam_epsilon if 'adam_epsilon' in config.keys() and isinstance(config.adam_epsilon, float) else 1e-8, type=float, help="Epsilon for Adam optimizer.")
    #     parser.add_argument("--max_grad_norm", default=config.max_grad_norm if 'max_grad_norm' in config.keys() and isinstance(config.max_grad_norm, float) else 1.0, type=float, help="Max gradient norm.")
        
    #     parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    #     parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    #     parser.add_argument('--logging_steps', type=int, default=1000, help="Log every X updates steps.")
    #     parser.add_argument('--save_steps', type=int, default=1000, help="Save checkpoint every X updates steps.")

    #     parser.add_argument("--do_train", action="store_true", default=True,help="Whether to run training.")
    #     parser.add_argument("--do_eval", action="store_true", default=True, help="Whether to run eval on the test set.")
    #     parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    #     args = parser.parse_args()

    #     args.model_name_or_path = MODEL_PATH_MAP[args.model_type]

    ################################3
    init_logger()
    set_seed(args)
    
    tokenizer = load_tokenizer(args)
    train_dataset = None
    dev_dataset = None
    test_dataset = None
    if args.do_train or args.do_eval:
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)
    # wandb.watch(trainer.model, criterion=None, log='all', log_freq=10)
    if args.do_train:
        trainer.train()
        # trainer.train(wandb)
    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test", "eval")

# def get_sweep_id(method):
#     sweep_config = {
#         'method': method
#         , 'metric' : {'name':'val_loss', 'goal':'minimize'}
#         , 'early_terminate' : {'type':'hyperband', 'min_iter':10, 'eta':3}
#         , 'parameters' : {
#             'model_type':{
#                 'values':['kobert','bert','koelectra-base']
#             },
#             'train_batch_size':{
#                 'values':[32]
#             },
#             'eval_batch_size':{
#                 'values':[64]
#             },
#             'max_seq_len':{
#                 'values':[50]
#             },
#             'learning_rate':{
#                 'values':[5e-5]
#             },
#             'num_train_epochs':{
#                 'values':[20.0]
#             },
#             'weight_decay':{
#                 'values':[0.0]
#             },
#             'gradient_accumulation_steps':{
#                 'values':[1]
#             },
#             'adam_epsilon':{
#                 'values':[1e-8]
#             },
#             'max_grad_norm':{
#                 'values':[1.0]
#             },
#             # 'max_steps':{
#             #     'values':[-1]
#             # },
#             'warmup_steps':{
#                 'values':[0]
#             },
#         }#, 'parameters' : {
#     }# sweep_config = {

#     sweep_id = wandb.sweep(sweep_config) 
#     return sweep_id


if __name__ == '__main__':

    # # 20210919
    # sweep_id = get_sweep_id('grid')
    # wandb.agent(sweep_id=sweep_id, function=main)
    
    

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="naver-ner", type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=f"{os.path.dirname(__file__)}/model", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default=f"{os.path.dirname(__file__)}/data", type=str, help="The input data dir")
    parser.add_argument("--pred_dir", default=f"{os.path.dirname(__file__)}/preds", type=str, help="The prediction file dir")

    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Slot Label file")

    parser.add_argument("--write_pred", default=True, action="store_true", help="Write prediction during evaluation")

    # parser.add_argument("--model_type", default="kobert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_type", default="koelectra-base", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    # parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    # parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--max_seq_len", default=256, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    # parser.add_argument('--logging_steps', type=int, default=512, help="Log every X updates steps.")
    parser.add_argument('--logging_steps', type=int, default=8, help="Log every X updates steps.")
    # parser.add_argument('--save_steps', type=int, default=512, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=8, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", default=True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
