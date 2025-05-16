# SAI3CNet: Symmetry-Aware Intelligent IIIC Classifier Network

A PyTorch implementation for automatically classifying epilepsy-related IIIC (Ictal-Interictal-Injury Continuum) patterns. This repository contains the SAI3CNet model, which combines LeadRelationshipEncoder (LRE) and Spectrogram Attention (SpAttn) Blocks, and training/evaluation scripts. The Self-Distilled EEG Learning Framework (SD-LF) proposed in the paper has not yet been coded, but the performance comparison results are included below.


# 📑 Dataset

The study used the Harmful Brain Activity in EEG (HBAC) dataset published by HMS, MassGeneralBrigham, and Harvard Medical School.

Kaggle link: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification

A total of 22,553 10-second EEG clips (20 leads)

6 classes: Periodic Discharges (PD), Rhythmic Delta Activity (RDA), Seizure, etc.


