# coding:utf8
from tqdm.auto import tqdm # for iter进度条
from torch.optim import AdamW
from transformers import get_scheduler

from E50_utils.E56_DatasetLoader import *
from transformers import GPT2ForSequenceClassification, AutoConfig

def train_once(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch):
    progress_bar = tqdm(range(len(dataloader)))

    finish_step_num = (epoch - 1) * len(dataloader)
    size = len(dataloader.dataset)
    total_loss, correct = 0, 0

    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(**X).logits # dim
        loss = loss_fn(pred, y)

        '''
        实际上，在这段代码中，optimizer.zero_grad()的位置并不会产生问题。这是因为在你的训练循环中，每次开始新的梯度计算之前，你都会清零梯度。这是在开始反向传播之前，也就是loss.backward()调用之前，就已经完成了。

        在PyTorch中，模型的梯度会在每次反向传播的时候累积，而不是被替换。这意味着如果你在反向传播之前没有将梯度清零，那么下一次的梯度计算就会和之前的梯度叠加，从而导致错误的结果。因此，为了确保正确的梯度计算，你需要在每次开始新的梯度计算之前将梯度清零。在你给出的代码中，这一点已经被很好地满足了。
        '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_loss += loss.item()

        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1)

    total_loss /= size
    correct /= size
    progress_bar.close()
    print(f'loss: {total_loss} accuracy: {correct}')

def evaluate_once(dataloader, model, loss_fn, mode='Valid'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    total_loss, correct = 0, 0
    tn, fp, fn, tp = 0, 0, 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(**X).logits
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            loss = loss_fn(pred, y)
            total_loss += loss.item()
            '''
            t/f:
                pred.argmax(1) == torch.max(pred, 1)[1] 即取一列，其中每个元素都是对应行的最大值 取索引
                pred.argmax(1) == y tensor([False,  True, False,  True,  True,  True,  True]) 
            p/n:
                y

            tf = (pred.argmax(1) == y).type(torch.int)
            pn = y

            -y+1 == ~y binary
            '''
            tf = (pred.argmax(1) == y).type(torch.int)

            tp += (tf * y).sum().item()
            tn += (tf * (-y + 1)).sum().item()
            fp += ((-tf + 1) * y).sum().item()
            fn += ((-tf + 1) * (-y + 1)).sum().item()

    correct /= size
    total_loss /= size

    print("== Valid Evaluate")
    print("Loss: ", total_loss)
    print(f"Accuracy: {100 * correct}")
    # print(f"{mode} Accuracy: {(100 * correct):>0.1f}%")
    print('False positive rate(FP): ', fp / (fp + tn) if (fp + tn) != 0 else 'NA')
    print('False negative rate(FN): ', fn / (fn + tp) if (fn + tp) != 0 else 'NA')
    recall = tp / (tp + fn) if (tp + fn) != 0 else 'NA'
    print('Recall: ', recall)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 'NA'
    print('Precision: ', precision)
    try:
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
    except:
        print('F1 score: ', 'NA')
    print()
    return correct

def train():
    # model load
    # config = AutoConfig.from_pretrained(checkpoint)
    if checkpoint=='':
        from transformers import GPT2Config
        config = GPT2Config()
        model = GPT2ForSequenceClassification(config=config).to(device)
        print('Loading model from random initial default')
    else:
        model  = GPT2ForSequenceClassification.from_pretrained(checkpoint).to(device)
        print('Loading model from:', checkpoint)

    # load data
    train_data, valid_data = load_data()
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
    train_data_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)
    valid_data_loader = DataLoader(valid_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)
    print('='*99,'\n')

    # train func
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
    lr_scheduler = get_scheduler(
        'linear'
        , optimizer=optimizer
        , num_warmup_steps=0
        , num_training_steps=conf.epoch_num * len(train_data_loader)
    )

    ## epoch loop
    last_acc = -1
    last_state = None
    for t in range(conf.epoch_num):
        print(f"Epoch {t + 1}/{conf.epoch_num}\n-------------------------------")
        train_once(train_data_loader, model, loss_fn, optimizer, lr_scheduler, t + 1)
        correct = evaluate_once(valid_data_loader, model, loss_fn, mode='Valid')
        if correct>=last_acc:
            last_acc = correct
            last_state = model.state_dict()
            print('save model acc:', correct)
        else:
            model.load_state_dict(last_state)
            print('load model acc:', last_acc)
    '''
        在这里保存一下结果last_state的整个模型结构
    '''
    save_name = conf.dataset.split('\\')[-1].split('/')[-1]
    save_name = f'checkpoint/for_dachuang/{save_name}_{last_acc}.pt'
    model.load_state_dict(last_state)
    torch.save(model, save_name)
    print('save model acc:', last_acc)
    print('save model name:', save_name)


if __name__ == '__main__':
    parse_arg()
    train()

