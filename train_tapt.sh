#!/usr/bin/env bash

# 2023-03-22
cp="microsoft/CodeGPT-small-java-adaptedGPT2"
tcp="gpt2"
tag="TAPT_codegpt.tokenized/TAPT_codegpt"
rt=1
for i in $(seq 1 1);
do
python E64_gpt2_tapt.py -lr=0.00005 -e=20 -rt=$rt -cp=$cp -tcp=$tcp -lmp=checkpoint/"$tag"_"$i"
done

cp="microsoft/CodeGPT-small-java-adaptedGPT2"
tcp="gpt2"
tag="TAPT_codegpt/TAPT_codegpt"
rt=0
for i in $(seq 1 1);
do
python E64_gpt2_tapt.py -lr=0.00005 -e=20 -rt=$rt -cp=$cp -tcp=$tcp -lmp=checkpoint/"$tag"_"$i"
done