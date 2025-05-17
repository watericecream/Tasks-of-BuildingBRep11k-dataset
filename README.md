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
```


## 2 Dataset

The dataset can be found at https://huggingface.co/datasets/WATERICECREAM/BuildingBRep11k. 

These are the details of each file.

| Filenname     | Description            |
|----------------------------|-------------------|
|BuildingBRep11k.tar|The mainbody of dataset, containing folder "3d_objects" with breps and  folder "images" with thumbnails|
|BuildingBRep11k/3d_objects|Subfolder, with all 11k .brep files in it|
|BuildingBRep11k/images|Subfolder, with all 11k breps' thumbnails in it.|
|meta.json|Detailed parameters of all 11k breps|
|meta.npy|Same content with the json file|
|Repro_t2.zip|Data used to reproduce the random sample in task2|
|Repro_t2/data|Subfolder, with data used for training the pth in task2.|
|Repro_t2/data_test|Subfolder, with random selected samples testing in task2.|

## 3 Run

Please make sure the environment is ready, then cd to the whole project's direction.

### 1 Task 1

```bash
cd Task1_mul_att_reg
```
Make a new folder "data" and copy all .step files into "data".

For testing the existed model "pointnet_multi.pth":

```bash
python test.py
```

The program will randomly select 100 samples from "data" and produce "predict_multi.txt" with predictions.

For training a new model:

```bash
python train.py
```

The program will start to train with randomly seleced 5000 samples recorded in "meta_5000.npy". If a new training with all 11k samples is required, just replace the existed .npy and .json files by those from huggingface, and replace the meta file name in dataset.py.

### 2 Task 2

```bash
cd Task2_def_det
```

Copy the two folders - "data" and "data_test" - from Repro_t2.zip into this directory. In this task, contents in "data" are for training and "data_test" for testing.

For testing the existed model "pre_train.pth":

```bash
python batch_inference_classify.py data_test/ pre_train.pth test_result.csv
```

The data directory, model and record can be user defined in line.

For training a new model:

```bash
python train_classify.py
```

The program will start to train with already selected 5000 good breps in "data/good/" and 5000 bad breps in "data/bad".

If more defected breps are needed, simply run 

```bash
python make_defect.py
```

don't forget to put the good breps neede to be made defect in the right folder.
