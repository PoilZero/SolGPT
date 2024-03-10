import torch
from torch.nn.functional import softmax
from transformers import GPT2Tokenizer
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
    code reading
'''
with open('test_code.sol', 'r', encoding='utf-8') as f:
    test_code = f.read()



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
X = tokenizer(
    [test_code],
    padding=True,
    truncation=True,
    return_tensors="pt"
)
dic_name = {
    'reentrancy': '可重入漏洞',
    'timestamp': '时间戳依赖漏洞',
    'delegatecall': '委托调用漏洞',
    'integeroverflow': '整数溢出漏洞',
}
dic01 = {0: '不存在该漏洞', 1: '存在该漏洞'}
cp_path = 'checkpoint/for_dachuang/'
for one in os.listdir(cp_path):
    print(dic_name[one.split('_')[0]], end=': ')
    cp_file = os.path.join(cp_path, one)
    model = torch.load(cp_file)
    X = X.to(device)
    model.to(device)
    model.eval()
    pred = softmax(model(**X).logits, dim=1)[0]
    print(dic01[int(pred.argmax(0))], pred)