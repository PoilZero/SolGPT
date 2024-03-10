# coding:utf8
import os

from torch.utils.data import Dataset
from E50_utils.E50_com import *

class SCVulData(Dataset):
    def __init__(self, data_file=None):
        self.lables = []
        self.data = self.load_data(data_file) if data_file!=None else self.load_data(conf.dataset)

    def load_data(self, data_file):
        Data = []
        idx = 0
        with open(data_file, 'rt', encoding='utf8') as f:
            fragment = []
            fragment_val = 0
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if "-" * 33 in line and fragment:
                    # yield fragment, fragment_val
                    sentence = '\n'.join(fragment)
                    # Data[idx] = {"sentence": sentence, "label": fragment_val}
                    # self.lables[idx] = fragment_val
                    Data.append({"sentence": sentence, "label": fragment_val})
                    # self.lables.append(fragment_val)
                    idx += 1
                    fragment = []
                elif stripped.split()[0].isdigit():
                    if fragment:
                        if stripped.isdigit():
                            fragment_val = int(stripped)
                        else:
                            fragment.append(stripped)
                else:
                    fragment.append(stripped)
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_label(self, idx):
        return self.data[idx]["label"]

rate = 0.7
def load_data(data_path=conf.dataset):
    if os.path.isdir(data_path):
        print('Loading folder data loaded')
        dir_path = data_path
        train_data = SCVulData(dir_path + '/train.txt')
        valid_data = SCVulData(dir_path + '/valid.txt')
    else:
        all_data = SCVulData(data_path)
        div = int(len(all_data) * rate)
        train_data, valid_data = all_data[:div], all_data[div:]
        print('Loading file data loaded with rate =', rate)
    return train_data, valid_data

if __name__== '__main__':
    train_data = SCVulData('../'+conf.dataset)
    print(train_data[0])
    print(len(train_data))

