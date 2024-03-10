# coding:utf-8
import os

import os

def handle(handle_path, label):
    print(f"Processing files in directory: {handle_path}")
    for file_name in os.listdir(handle_path):
        file_path = os.path.join(handle_path, file_name)
        print(f"Processing file: {file_path}")
        with open(file_path, 'rt', encoding='utf8') as f:
            segments = f.read().strip().split('-'*33)
        new_segments = []
        for segment in segments:
            if not segment.strip():
                continue
            new_segment = segment.strip().split('\n')
            if new_segment[-1].strip()!='0':
                new_segment[-1] = str(label)
            new_segment = '\n'.join(new_segment)
            new_segments.append(new_segment)
        tag = '-'*33
        new_segments = f'\n{tag}\n'.join(new_segments)
        print(f"Writing to file: {file_path}")
        with open(file_path, 'wt', encoding='utf8') as f:
            f.write(new_segments)


task_dic = {'./divided/delegatecall_196_directlly_from_dataset':'3', './divided/timestamp_349_directlly_from_dataset':'2', './divided/integeroverflow_275_directlly_from_dataset':'4', './divided/reentrancy_273_directlly_from_dataset':'1'}

for handle_path, label in task_dic.items():
    handle(handle_path, label)
    print()



