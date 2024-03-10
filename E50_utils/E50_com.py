# coding:utf8
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# checkpoint_bert = 'microsoft/codebert-base' # 127M
# checkpoint_gpt2 = 'microsoft/CodeGPT-small-java-adaptedGPT2' # 128M
# checkpoint = "checkpoint/xblock_combined_javagpt2/cp_1_0.33431644455176496"
checkpoint = 'gpt2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# hyper parameter
class Config():
    def __init__(self):
        self.dataset    = r'D:\__dachuang2022\C20_SolGPT\dataset\v3_73\FinetuningB\reentrancy_273_directlly_from_dataset'
        self.learning_rate = 0.00005 # dim
        self.epoch_num     = 50
        self.batch_size    = 4
        self.retrain_tokenizer = 0
        self.lm_path = 'checkpoint/TAPT'
        self.tokenizer_checkpoint = checkpoint
conf = Config()

def parse_arg():
    # conf.dataset = sys.argv[1] if len(sys.argv)>=2 else None
    global checkpoint
    parser = argparse.ArgumentParser(description='SCVulBert for Smart Contracts Vul Detection')
    parser.add_argument(
        'dataset', nargs='?'
        , default=conf.dataset
    )
    parser.add_argument(
        '--learning_rate', '-lr'
        , type=float, default=conf.learning_rate
    )
    parser.add_argument(
        '--epoch_num', '-e'
        , type=int, default=conf.epoch_num
    )
    parser.add_argument(
        '--batch_size', '-bs'
        , type=int, default=conf.batch_size
    )
    parser.add_argument(
        '--retrain_tokenizer', '-rt'
        , type=int, default=conf.retrain_tokenizer
    )
    parser.add_argument(
        '--checkpoint', '-cp'
        , type=str, default=checkpoint
    )
    parser.add_argument(
        '--tokenizer_checkpoint', '-tcp'
        , type=str, default=conf.tokenizer_checkpoint
    )
    parser.add_argument(
        '--lm_path', '-lmp'
        , type=str, default=conf.lm_path
    )
    args = parser.parse_args()
    print('Using HyperParameter as follows:')
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg=='checkpoint':
            checkpoint = arg_value
            print('- checkpoint =', checkpoint)
            continue
        else:
            print('-', arg, '=', arg_value)
            setattr(conf, arg, arg_value)
parse_arg()