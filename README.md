# Tasks-of-BuildingBRep11k-dataset

These are the 2 baseline codes, and are tested on Windows system.

Please create a new conda environment to run the two tasks.

First create and activate the environment.

conda create -n build11k python=3.10 -y

conda activate build11k

Then install packages:

pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install torchmetrics==1.3.2 open3d pandas tqdm

conda install -c conda-forge -c cadquery ocp

Then you can find the data for training and testing here: https://huggingface.co/datasets/WATERICECREAM/BuildingBRep11k/tree/main

# BuildingBRep-11K ∙ Baseline Tasks

This repository contains **two lightweight baselines** (PointNet-based) for
BuildingBRep-11K:

1. **Multi-attribute regression** – predict storey count, room totals, etc.  
2. **Defect classification** – label a B-rep as *GOOD* or *DEFECT*.

The scripts were developed and tested on **Windows 10 / 11** with **CUDA 12.1**.
A clean **conda** environment is recommended.

---

## 1  Quick Start

```bash
# ❶ create & activate
conda create -n build11k python=3.10 -y
conda activate build11k

# ❷ install PyTorch (CUDA 12.1 wheels)
pip install torch==2.3.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ❸ install remaining Python deps
pip install torchmetrics==1.3.2 open3d pandas tqdm

# ❹ install OpenCASCADE (OCP) bindings
conda install -c conda-forge -c cadquery ocp
