import json

from E50_utils.E56_DatasetLoader import *

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch, random, os
from tqdm.auto import tqdm # for iter进度条

def select_top_k(predictions, k=10):
    # predictions(bs, seq_len, 50257/dic, feature)
    # => last_token_predictions(50257/dic, feature)
    last_token_predictions = predictions[0, -1, :]
    # (50257/dic, feature)=>((50257/dic, feature), (50257/dic, index))
    feature, feature_index = last_token_predictions.sort(descending=True)

    predicted_index = random.choice(feature_index[:10]).item()
    return predicted_index

def LMload_data(model):
    tokenizer = load_tokenizer(model)

    # load data
    train_data, valid_data = load_data()
    def SCVul_collate_fn(batch_samples):
        '''batch_samples=={"sentence": sentence, "label": fragment_val} for _ in range(batch_size)'''
        batch_sentences = [one['sentence'] for one in batch_samples]
        X = tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt" # torch.tensor
        )
        Y = X.copy()
        return X, Y
    train_data_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)
    valid_data_loader = DataLoader(valid_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)
    print('='*99,'\n')
    return train_data_loader, valid_data_loader

def save_model(model, cp_name):
    if not os.path.isdir(cp_name):
        os.makedirs(cp_name)
    model.save_pretrained(cp_name)
    print('saved model to', cp_name)
def LMload_model(cp_path=conf.lm_path):
    os.mkdir(cp_path) if not os.path.isdir(cp_path) else None

    filenames = os.listdir(cp_path)
    filenames.sort(reverse=True)
    if len(filenames):
        '''
            文件名的规则是 cp_th_0.xxxx
            要找到th最大的那一个作为target_filename
        '''
        target_filename = None
        max_th = -1
        for one in filenames:
            if not one.startswith('cp_'):
                continue
            th = one.split('_')[1]
            th = int(th)
            if th > max_th:
                max_th = th
                target_filename = one

        cp = os.path.join(cp_path, target_filename)
        start_epoch = int(target_filename.split('_')[1]) + 1
    else:
        cp = checkpoint
        start_epoch = 0

    print('Loading model from:', cp, 'start_epoch:', start_epoch)
    model = GPT2LMHeadModel.from_pretrained(cp).to(device)
    return model, start_epoch

def train_once(train_data_loader, model, optimizer):
    model.train_fedavg()
    progress_bar = tqdm(range(len(train_data_loader)))
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        # {'loss': loss, 'logits': logits, 'past_key_values': past_key_values}
        data, target = data['input_ids'], target['input_ids']
        output = model(data, labels=target)
        loss = output.loss

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        cur_mean_loss = total_loss / (batch_idx + 1)
        progress_bar.set_description(f'loss: {cur_mean_loss:>7f}')
        progress_bar.update(1)

    progress_bar.close()
    last_mean_loss = total_loss / len(train_data_loader)
    print('average loss:', last_mean_loss)

def evaluate_once(dataloader, model, mode='Valid'):
    assert mode in ['Valid', 'Test']

    model.eval()
    with torch.no_grad():
        print("== Valid Evaluate")
        progress_bar = tqdm(range(len(dataloader)))
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # {'loss': loss, 'logits': logits, 'past_key_values': past_key_values}
            data, target = data['input_ids'], target['input_ids']
            output = model(data, labels=target)
            loss = output.loss

            # loss caculate
            total_loss += loss.item()
            cur_mean_loss = total_loss / (batch_idx + 1)
            progress_bar.set_description(f'loss: {cur_mean_loss:>7f}')
            progress_bar.update(1)

        progress_bar.close()
        last_mean_loss = total_loss / len(dataloader)
        print('average loss:', last_mean_loss)

    return last_mean_loss


def train():
    model, start_epoch = LMload_model()
    train_data_loader, valid_data_loader = LMload_data(model)

    last_loss = 999
    last_state = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.learning_rate)  # 定义优化器
    for i in range(start_epoch, conf.epoch_num):
        print(f"Epoch {i + 1}/{conf.epoch_num}\n-------------------------------")
        train_once(train_data_loader, model, optimizer)
        train_loss = evaluate_once(valid_data_loader, model, mode='Valid')

        cp_name = f'{conf.lm_path}/cp_{i}_{train_loss}'
        if train_loss<=last_loss:
            last_loss = train_loss
            last_state = model.state_dict()
            print('save model train loss:', train_loss)
            save_model(model, cp_name)
        else:
            model.load_state_dict(last_state)
            print('load model train loss:', last_loss)

train()