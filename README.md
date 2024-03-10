# SolGPT

Model Training Part for Solidity vulnerability Detection based on GPT (torch backen)

## Paper

Paper "SolGPT: a GPT-Based Static Vulnerability Detection Model for Enhancing Smart Contract Security" has published in ICA3PP 2023.

Author: Shengqiang Zeng, Hongwei Zhang, Jinsong Wang, and Kai Shi.

Paper Manuscript already upload as ["SolGPT.9.17.2023"](https://github.com/PoilZero/SolGPT/blob/main/SolGPT.9.17.2023.pdf).

## Requirements

### Required Packages

* python3
* torch1（https://pytorch.org/get-started/locally/）
* transformers

### Required Dataset

This repository contains the smart contract dataset of source code and processed code fragment. As to the source code, we crawled the source code of the smart contract from the [Ethereum](https://etherscan.io/) by using the crawler tool. Besides, we have collected some data from some other websites. At the same time, we also designed and wrote some smart contract codes with reentrancy vulnerabilities.

**Note:** crawler tool is available [here](https://github.com/Messi-Q/Crawler).

### Dataset

Original Ethereum smart contracts: [Etherscan_contract](https://drive.google.com/open?id=1h9aFFSsL7mK4NmVJd4So7IJlFj9u0HRv)

The train data after normalization in `dataset`.

## OverView

### Trainning

Training contain two part
* TAPT training
* finetunnng

You can see the instruction in train_tapt.sh and train_finetune.sh to learn how to train this model

## References

* Rechecker for Normalized dataset: https://github.com/Messi-Q/ReChecker
  * This Project already stops maintained, this is a branch that I keep maintaining.
  * https://github.com/PoilZero/ReChecker
