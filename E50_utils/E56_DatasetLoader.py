# coding:utf8
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification

from E50_utils.E55_Dataset import *

def test_tokenize(tokenizer):
    sc_tokens = [
        'abstract', 'after', 'alias', 'allow', 'anonymous', 'apply', 'approve', 'assert', 'async', 'await',
        'begin', 'block', 'body', 'bool', 'break', 'byte', 'bytes', 'calldata', 'case', 'catch', 'chain',
        'class', 'clean', 'clone', 'const', 'constructor', 'continue', 'contract', 'copy', 'declare',
        'default', 'define', 'delegate', 'delete', 'depth', 'derive', 'desc', 'detach', 'do', 'else', 'emit',
        'encoding', 'end', 'enum', 'error', 'event', 'external', 'extract', 'factory', 'false', 'fast',
        'final', 'fixed', 'for', 'from', 'function', 'get', 'globals', 'goto', 'group', 'hash', 'if',
        'implements', 'import', 'indexed', 'inherited', 'init', 'inline', 'input', 'instanceof', 'int',
        'interface', 'internal', 'is', 'iterator', 'library', 'let', 'limit', 'lock', 'log', 'loop', 'map',
        'match', 'memory', 'modifier', 'module', 'mutable', 'namespace', 'native', 'new', 'none', 'null',
        'of', 'off', 'on', 'open', 'operator', 'optimize', 'or', 'override', 'pack', 'pragma', 'private',
        'promise', 'protected', 'public', 'pure', 'query', 'raise', 'readonly', 'rebase', 'receive', 'record',
        'reference', 'register', 'relocatable', 'remove', 'rename', 'repeat', 'replace', 'require', 'reset',
        'resolve', 'revert', 'returns', 'rollback', 'self', 'set', 'shl', 'shr', 'sizeof', 'stack', 'state',
        'static', 'stop', 'storage', 'string', 'struct', 'submit', 'substitute', 'success', 'super', 'switch',
        'synchronized', 'syntax', 'tag', 'test', 'then', 'throw', 'timeout', 'timestamp', 'to', 'transaction',
        'true', 'try', 'type', 'typeof', 'uint', 'unchecked', 'undelegate', 'unicode', 'unique', 'unpack',
        'unsafe', 'untrusted', 'update', 'use', 'using', 'var', 'view', 'virtual', 'visible', 'void',
        'volatile', 'wait', 'while', 'with', 'witness', 'xor'
    ]
    new_tokens = []
    for one in sc_tokens:
        if len(tokenizer.tokenize(one))!=1: # 专业词被切分
            new_tokens.append(one)
    print('new_tokens =', new_tokens)
    return new_tokens

def posttrain_tokenizer(model, tokenizer):
    if conf.retrain_tokenizer==1:
        new_tokens = test_tokenize(tokenizer)

        # tmp = []
        # for one in new_tokens:
        #     tmp.append(tokenizer.tokenize(one))
        # # print(tmp)
        # # print()
        # tmp = []
        # for one in new_tokens:
        #     tmp.append(tokenizer.encode(one))
        # # print(tmp)
        # # print()

        num_added_toks = tokenizer.add_tokens(new_tokens)  # 返回一个数，表示加入的新词数量，在这里是2
        print(f'post-training tokenizer with {num_added_toks} new') # 52
        # print()
        # tmp = []
        # for one in new_tokens:
        #     tmp.append(tokenizer.tokenize(one))
        # print(tmp)
        # print()
        # tmp = []
        # for one in new_tokens:
        #     tmp.append(tokenizer.encode(one))
        # print(tmp)
        # print()
    else:
        print(f'post-training tokenizer are turned off')

    # 关键步骤，resize_token_embeddings输入的参数是tokenizer的新长度
    model.resize_token_embeddings(len(tokenizer))
    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

def load_tokenizer(model, cp_path=conf.tokenizer_checkpoint):
    # if len(checkpoint.split('_'))<2:
    #     print('online checkpoint', checkpoint)
    #     tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    #     posttrain_tokenizer(model, tokenizer)
    #     tokenizer.save_pretrained(cp_path)
    # elif os.path.isdir(cp_path):
    #     print('load tokenizer from', cp_path)
    #     tokenizer = GPT2Tokenizer.from_pretrained(cp_path)
    # else:
    #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #     posttrain_tokenizer(model, tokenizer)
    #     tokenizer.save_pretrained(cp_path)
    #     print('save tokenizer to', cp_path)
    print('Loading tokenizer from:', cp_path)
    tokenizer = GPT2Tokenizer.from_pretrained(cp_path)
    posttrain_tokenizer(model, tokenizer)

    # default to left padding
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

'''
    :return tokenlized dataset, tensorlized label
'''

if __name__ == "__main__":
    config = AutoConfig.from_pretrained(checkpoint)
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=config).to(device)
    tokenizer = load_tokenizer(model)

    def SCVul_collate_fn(batch_samples):
        batch_sentence = []
        batch_label = []
        for sample in batch_samples:
            batch_sentence.append(sample['sentence'])
            batch_label.append(int(sample['label']))
        X = tokenizer(
            batch_sentence,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        y = torch.tensor(batch_label)
        return X, y
    train_data = SCVulData('../dataset/reentrancy_273_directlly_from_dataset.txt')
    train_data_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)

    batch_X, batch_y = next(iter(train_data_loader))
    print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    print('batch_y shape:', batch_y.shape)

    # print(train_data_loader.dataset==train_data)