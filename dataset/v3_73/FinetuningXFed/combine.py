# coding:utf-8
import os

for mode in ['train.txt', 'valid.txt']:
    print(f"Processing files in directory: {'./divided'}")
    combined_new_segments = []
    for file_name in os.listdir('./divided'):
        file_path = os.path.join('./divided', file_name)
        file_path = os.path.join(file_path, mode)
        print(f" - Processing file: {file_path}")
        with open(file_path, 'rt', encoding='utf8') as f:
            segments = f.read().strip().split('-'*33)
        new_segments = []
        for segment in segments:
            if not segment.strip():
                continue
            new_segments.append(segment.strip())

        combined_new_segments.extend(new_segments)

    print(' - Size', len(combined_new_segments))
    tag = '-' * 33
    combined_new_segments = f'\n{tag}\n'.join(combined_new_segments)

    w_path = os.path.join('./combined', f'{mode}')
    print(f" - Writing to file: {w_path}")
    with open(w_path, 'wt', encoding='utf8') as f:
        f.write(combined_new_segments)