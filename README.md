# SAI3CNet: Symmetry-Aware Intelligent IIIC Classifier Network

A PyTorch implementation for automatically classifying epilepsy-related IIIC (Ictal-Interictal-Injury Continuum) patterns. This repository contains the SAI3CNet model, which combines LeadRelationshipEncoder (LRE) and Spectrogram Attention (SpAttn) Blocks, and training/evaluation scripts. The Self-Distilled EEG Learning Framework (SD-LF) proposed in the paper has not yet been coded, but the performance comparison results are included below.


## 📑 Dataset

The study used the Harmful Brain Activity in EEG (HBAC) dataset published by HMS, MassGeneralBrigham, and Harvard Medical School.

Kaggle link: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification

A total of 22,553 10-second EEG clips (20 leads)

6 classes: Periodic Discharges (PD), Rhythmic Delta Activity (RDA), Seizure, etc.

##⚡ Quick Start

```bash
# 1. 저장소 클론 및 환경 설정
$ git clone https://github.com/your-org/sai3cnet.git
$ cd sai3cnet
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt


## 📊 Results

### 2. Ablation Study

| Model | ACC | F1 | KLD&nbsp;↓ | Sensitivity | mACC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EEG-based 1D CNN** |  |  |  |  |  |
| SPaRCNet            | 0.636 | 0.546 | 0.698 | 0.447 | 0.511 |
| EEG Conformer    | 0.351 | 0.298 | 0.869 | 0.337 | 0.341 |
| 1D GNN-CNN        | 0.394 | 0.246 | 0.869 | 0.192 | 0.248 |
| **Spectrogram-based 2D CNN (ImageNet pre-trained)** |  |  |  |  |  |
| ResNet-34          | 0.647 | 0.570 | 0.679 | 0.553 | 0.559 |
| DenseNet-121       | 0.659 | 0.584 | 0.677 | 0.585 | 0.576 |
| **Ours** |  |  |  |  |  |
| SAI3CNet                | 0.730 | 0.669 | 0.638 | 0.661 | 0.647 |
| **SAI3CNet + SD-LF**    | **0.747** | **0.705** | **0.625** | **0.700** | **0.699** |


### 2. Ablation Study

| Method | ACC | F1 | KLD&nbsp;↓ | Sensitivity | mACC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SAI3CNet                       | 0.735 | 0.673 | 0.627 | 0.666 | 0.657 |
| SAI3CNet w/ SpAttn             | 0.730 | 0.669 | 0.638 | 0.661 | 0.647 |
| SAI3CNet w/o SpAttn & LRE      | 0.728 | 0.644 | 0.670 | 0.644 | 0.629 |
> ⬆️ **SD-LF** code is currently not publicly available. The above values are reproduced from the paper.

