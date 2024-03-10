from tqdm.auto import tqdm # for iter进度条
from torch.optim import AdamW
from transformers import get_scheduler

from E50_utils.E56_DatasetLoader import *
from transformers import GPT2ForSequenceClassification, AutoConfig

def create_model():
    config = AutoConfig.from_pretrained(checkpoint, num_labels=5)
    global_model = GPT2ForSequenceClassification.from_pretrained(checkpoint, config=config).to(device)
    tokenizer = load_tokenizer(global_model)  # 为了初始化模型的输入词表
    return global_model, tokenizer

def federated_averaging(participants, global_model):
    # 显存优化
    global_model.to('cpu')

    # 加权系数分母
    coefficient_sum = 0
    for participant in participants:
        coefficient_sum += participant.weight

    for name, param in global_model.named_parameters():
        # 个性化不更新分类头
        if name=='score.weight':
            continue

        # 初始化权重的累积和为0
        weight_sum = torch.zeros_like(param.data)

        # 遍历每个参与者的模型加权权重并相加
        for participant in participants:
            coefficient =  1.0 # participant.weight
            weight_sum += coefficient*participant.model.state_dict()[name]

        # 计算平均权重
        avg_weight = weight_sum / len(participants)

        # 将新计算的平均权重设置为全局模型的权重
        param.data = avg_weight

    # 显存优化
    global_model.to(device)

    return global_model

class Paticipant():
    def __init__(self, model, train_dataloader, valid_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
        self.lr_scheduler = get_scheduler(
            'linear'
            , optimizer=self.optimizer
            , num_warmup_steps=0
            , num_training_steps=conf.epoch_num * len(self.train_dataloader)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.weight  = len(train_dataloader.dataset) + len(valid_dataloader.dataset)

    def train_once(self, epoch):
        # 显存优化
        self.model.to(device)

        dataloader = self.train_dataloader
        model = self.model
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        loss_fn = self.loss_fn

        progress_bar = tqdm(range(len(dataloader)))

        finish_step_num = (epoch - 1) * len(dataloader)
        size = len(dataloader.dataset)
        total_loss, correct = 0, 0

        model.train()
        for step, (X, y) in enumerate(dataloader, start=1):
            X, y = X.to(device), y.to(device)
            pred = model(**X).logits # dim
            loss = loss_fn(pred, y)

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

        # 显存优化
        self.model.to('cpu')

    def evaluate_once(self, mode = 'Valid'):
        # 显存优化
        self.model.to(device)

        dataloader = self.valid_dataloader
        model = self.model
        loss_fn = self.loss_fn

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
        # # print(f"{mode} Accuracy: {(100 * correct):>0.1f}%")
        # print('False positive rate(FP): ', fp / (fp + tn) if (fp + tn) != 0 else 'NA')
        # print('False negative rate(FN): ', fn / (fn + tp) if (fn + tp) != 0 else 'NA')
        # recall = tp / (tp + fn) if (tp + fn) != 0 else 'NA'
        # print('Recall: ', recall)
        # precision = tp / (tp + fp) if (tp + fp) != 0 else 'NA'
        # print('Precision: ', precision)
        # try:
        #     print('F1 score: ', (2 * precision * recall) / (precision + recall))
        # except:
        #     print('F1 score: ', 'NA')
        # print()

        # 显存优化
        self.model.to('cpu')
        return correct

    def sync_model(self, state_dict):
        # self.model.load_state_dict(state_dict)
        # 个性化学习不更新分类头
        model_dict = self.model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k != 'score.weight'}
        model_dict.update(filtered_state_dict)
        self.model.load_state_dict(model_dict)

import numpy as np
from torch.utils.data import Subset


class NonIIDDatasetCreator:
    def __init__(self, num_clients, alpha):
        self.num_clients = num_clients
        self.alpha = alpha

    def create_non_iid_dataset(self, dataset):
        num_samples = len(dataset)
        num_classes = len(set(dataset.get_label(idx) for idx in range(num_samples)))

        # 首先按类别将数据集划分为子集
        subsets = [[] for _ in range(num_classes)]
        for idx in range(num_samples):
            label = dataset.get_label(idx)
            subsets[label].append(idx)

        non_iid_datasets = [[] for _ in range(self.num_clients)]
        for subset in subsets:
            num_samples_subset = len(subset)

            # 使用Dirichlet分布生成每个样本的数据分布
            sample_distribution = np.random.dirichlet(np.repeat(self.alpha, num_samples_subset))

            remaining_indices = list(range(num_samples_subset))  # 记录剩余的未选中的样本的索引
            for i in range(self.num_clients):
                # 计算每个客户端应该获取的样本数量
                samples_per_client = int(num_samples_subset / self.num_clients)
                client_samples = min(samples_per_client, len(remaining_indices))

                # 根据样本的数据分布随机选择样本
                chosen_indices = np.random.choice(remaining_indices, size=client_samples, replace=False, p=sample_distribution)

                # 添加到每个客户端的非IID数据集
                non_iid_datasets[i].extend([subset[idx] for idx in chosen_indices])

                # 从分布和remaining_indices中移除已经选中的样本
                for chosen_index in sorted(chosen_indices, reverse=True):
                    actual_index = remaining_indices.index(chosen_index)
                    del remaining_indices[actual_index]
                    sample_distribution = np.delete(sample_distribution, actual_index)

                if len(sample_distribution) > 0:  # 避免在没有样本可选时进行归一化
                    sample_distribution /= sample_distribution.sum()  # 重新归一化

        # 创建非IID的子数据集
        non_iid_datasets = [Subset(dataset, indices) for indices in non_iid_datasets]

        return non_iid_datasets





# # 示例用法
# # 假设你有一个名为dataset的PyTorch数据集
# # 假设数据集中共有5个类别
#
# num_clients = 10  # 客户端数量
# alpha = 0.5  # Dirichlet分布的参数alpha
# num_classes = 5  # 样本的类别数量
#
# dataset_creator = NonIIDDatasetCreator(num_clients, alpha, num_classes)
# non_iid_datasets = dataset_creator.create_non_iid_dataset(dataset)
#
# # 创建每个客户端的DataLoader
# batch_size = 32  # 批次大小
#
# client_dataloaders = []
# for non_iid_dataset in non_iid_datasets:
#     dataloader = DataLoader(non_iid_dataset, batch_size=batch_size, shuffle=True)
#     client_dataloaders.append(dataloader)

'''
    通过non-iid划分重新分布数据集
    构建train_datasets, valid_datasets
'''
num_clients = 10  # 客户端数量
alpha = 0.5  # Dirichlet分布的参数alpha
num_classes = 5  # 样本的类别数量
dataset_creator = NonIIDDatasetCreator(num_clients, alpha)

train_data, valid_data = load_data(conf.dataset)
non_iid_train_datas = dataset_creator.create_non_iid_dataset(train_data)
non_iid_valid_datas = dataset_creator.create_non_iid_dataset(valid_data)

participants = []
data_file_map = []
# # divided_path = os.path.join(conf.dataset, 'divided')
# divided_path = conf.dataset
# for data_file in os.listdir(divided_path):
#     data_path = os.path.join(divided_path, data_file)
#     train_data, valid_data = load_data(data_path)
#
#     if not(os.path.isdir(data_path)):
#         continue
for i in range(len(non_iid_train_datas)):
    train_data = non_iid_train_datas[i]
    valid_data = non_iid_valid_datas[i]

    model, tokenizer = create_model()
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
    participants.append(Paticipant(model, train_data_loader, valid_data_loader))
    data_file_map.append('non_iid')

# '''
#     LOAD PARTICIPANTs in data loop
# '''
# participants = []
# data_file_map = []
# # divided_path = os.path.join(conf.dataset, 'divided')
# divided_path = conf.dataset
# for data_file in os.listdir(divided_path):
#     data_path = os.path.join(divided_path, data_file)
#     train_data, valid_data = load_data(data_path)
#
#     if not(os.path.isdir(data_path)):
#         continue
#
#     model, tokenizer = create_model()
#     def SCVul_collate_fn(batch_samples):
#         batch_sentence = []
#         batch_label = []
#         for sample in batch_samples:
#             batch_sentence.append(sample['sentence'])
#             batch_label.append(int(sample['label']))
#         X = tokenizer(
#             batch_sentence,
#             padding=True,
#             truncation=True,
#             return_tensors="pt"
#         )
#         y = torch.tensor(batch_label)
#         return X, y
#     train_data_loader = DataLoader(train_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)
#     valid_data_loader = DataLoader(valid_data, batch_size=conf.batch_size, shuffle=True, collate_fn=SCVul_collate_fn)
#     participants.append(Paticipant(model, train_data_loader, valid_data_loader))
#     data_file_map.append(data_file)

'''
    training
'''
global_model,tokenizer = create_model()
for epoch in range(1, conf.epoch_num+1):
    print('-'*33)
    print('= Fed Learning Epoch %d =' % epoch)
    print('-'*33)
    for i in range(len(participants)):
        print('== Participant %d ==' % i, data_file_map[i])
        participant = participants[i]

        participant.sync_model(global_model.state_dict())
        participant.evaluate_once()
        participant.train_once(epoch)
        participant.evaluate_once()
        print()
    global_model = federated_averaging(participants, global_model)
    print()