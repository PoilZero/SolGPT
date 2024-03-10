# coding:utf-8
import os

import os

def evaluate(handle_path):
    # print(f"Processing files in directory: {handle_path}")
    dataset_pn = 0
    dataset_nn = 0
    for file_name in os.listdir(handle_path):
        cnt = 0
        dic = {}
        file_path = os.path.join(handle_path, file_name)
        # print(f"Processing file: {file_path}")
        with open(file_path, 'rt', encoding='utf8') as f:
            segments = f.read().strip().split('-'*33)
        new_segments = []
        for segment in segments:
            if not segment.strip():
                continue
            new_segment = segment.strip().split('\n')
            # print(new_segment[-1].strip())
            lab = new_segment[-1].strip()
            dic[lab] = dic.get(lab, 0)+1
            if new_segment[-1].strip()!='0':
                new_segment[-1] = str(label)
                new_segment[-1].strip()
                cnt += 1
            new_segment = '\n'.join(new_segment)
            new_segments.append(new_segment)
        mode = file_name.strip().split('.')[-2]
        '''
        train set rate pos/all=0.3229166666666667 (pos: 62, neg: 130, all: 192)
        valid set rate pos/all=0.3373493975903614 (pos: 28, neg: 55, all: 83)
        '''
        pn = cnt
        nn = len(new_segments)-cnt
        dataset_pn += pn
        dataset_nn += nn
        # print(f'{mode} set rate pos/all={cnt/len(new_segments)} (pos: {cnt}, neg: {len(new_segments)-cnt}, all: {len(new_segments)})')

    # 例子 ./reentrancy_273_directlly_from_dataset 截取成 reentrancy 输出
    print(f'vul name: {handle_path.split("/")[-1].split("_")[0]}')
    print(f'data set rate pos/all={dataset_pn/(dataset_pn+dataset_nn)} (pos: {dataset_pn}, neg: {dataset_nn}, all: {dataset_pn+dataset_nn})')

task_dic = {'./delegatecall_196_directlly_from_dataset':'3'
    , './timestamp_349_directlly_from_dataset':'2'
    , './integeroverflow_275_directlly_from_dataset':'4'
    , './reentrancy_273_directlly_from_dataset':'1'
}

for handle_path, label in task_dic.items():
    evaluate(handle_path)
    print()



