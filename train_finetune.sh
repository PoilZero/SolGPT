#!/usr/bin/env bash

#############################################################################
cp="checkpoint/codetokenizer/TAPT_codegpt_1/cp_4_0.8440246127727555"
tcp="microsoft/CodeGPT-small-java-adaptedGPT2"
rt=0
log_path="codetokenizer.TAPT_codegpt"
for i in $(seq 1 5);
do
python E65_train_gpt2.py dataset/v3_73/FinetuningB/integeroverflow_275_directlly_from_dataset -lr=0.00005 -rt=$rt -e=50 -cp=$cp -tcp=$tcp | tee logs/$log_path/integeroverflow_275_directlly_from_dataset.txt_"$i".log;
python E65_train_gpt2.py dataset/v3_73/FinetuningB/delegatecall_196_directlly_from_dataset -lr=0.00005 -rt=$rt -e=50 -cp=$cp -tcp=$tcp | tee logs/$log_path/delegatecall_196_directlly_from_dataset.txt_"$i".log;
python E65_train_gpt2.py dataset/v3_73/FinetuningB/timestamp_349_directlly_from_dataset -lr=0.00005 -rt=$rt -e=50 -cp=$cp -tcp=$tcp | tee logs/$log_path/timestamp_349_directlly_from_dataset.txt_"$i".log;
python E65_train_gpt2.py dataset/v3_73/FinetuningB/reentrancy_273_directlly_from_dataset -lr=0.00005 -rt=$rt -e=50 -cp=$cp -tcp=$tcp | tee logs/$log_path/reentrancy_273_directlly_from_dataset.txt_"$i".log;
done
#############################################################################