# dataset_classify.py — Good/Defect 二分类数据集
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

from utils import load_step, sample_shape, normalise

class DefectDataset(Dataset):
    def __init__(self, root='data', split='train', n_pts=4000, seed=0):
        """
        root:     data 根目录
        split:    'train'|'val'|'test'
        n_pts:    每个样本采点数量
        """
        good = sorted(Path(root, 'good').glob('*.step'))
        bad  = sorted(Path(root, 'bad' ).glob('*.step'))
        # 标签 0=Good, 1=Defect
        all_files = [(p,0) for p in good] + [(p,1) for p in bad]

        # 打乱、划分
        rng = np.random.RandomState(seed)
        rng.shuffle(all_files)
        n = len(all_files)
        n_train = max(1, int(0.8 * n))
        n_val   = max(1, int(0.1 * n))
        if split == 'train':
            sel = all_files[:n_train]
        elif split == 'val':
            sel = all_files[n_train:n_train+n_val]
        else:
            sel = all_files[n_train+n_val:]
        self.samples = sel
        self.n_pts    = n_pts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # 1) 读 STEP → 点云
        shape = load_step(path)
        pts   = sample_shape(shape, n_pts=self.n_pts)
        pts   = normalise(pts)
        # 2) 返回 (B,N,3) 里的一条： (N,3), label:int
        return torch.from_numpy(pts), torch.tensor(label, dtype=torch.long)
