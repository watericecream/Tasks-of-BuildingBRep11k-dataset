# dataset.py  ── loads 100-sample subset
import pathlib, random, torch, numpy as np, json
from utils import load_step, sample_shape

DATA_DIR   = pathlib.Path("data")
META_NPY   = pathlib.Path("meta_5000.npy")
META_JSON  = pathlib.Path("meta_5000.json")

# ---------- load metadata ----------
if META_NPY.exists():
    raw = np.load(META_NPY, allow_pickle=True).item()
elif META_JSON.exists():
    with open(META_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
else:
    raise FileNotFoundError("meta.npy / meta.json not found")

META = {int(k): v for k, v in raw.items()}      # ensure int keys

def normalise(pc):
    pc -= pc.mean(0)
    pc /= np.linalg.norm(pc, axis=1).max()
    pc[:, 2] += random.uniform(-0.3, 0.3)       # kill absolute height cue
    return pc

class BuildingDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", seed=0):
        random.seed(seed)
        ids = sorted(META.keys())               # here: 1-100
        random.shuffle(ids)
        n = len(ids)
        if split == "train": ids = ids[:int(.8*n)]       # 80
        elif split == "val": ids = ids[int(.8*n):int(.9*n)]  # 10
        else:                ids = ids[int(.9*n):]       # 10
        self.ids = ids

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        bid = self.ids[idx]
        pc  = sample_shape(load_step(DATA_DIR/f"{bid}.step"), 2048)
        pc  = normalise(pc)
        md = META[bid]  # 楼栋元数据 dict

        # --- ① 楼层数 (9-类) -----------------
        storey_cls = md['storeys'] - 2  # 0–8

        # --- ② 总房间数 ----------------------
        room_tot = 0
        room_per = torch.zeros(10, dtype=torch.float32)  # F1…F9

        for i in range(1, md['storeys'] + 1):
            n = len(md.get(f'centroid_F{i}', []))
            room_tot += n
            room_per[i - 1] = n  # 多余层保持 0

        # --- ③ 平均面积 ----------------------
        areas = []

        for i in range(1, md['storeys'] + 1):

            for w, h in md.get(f'wh_F{i}', []):

                areas.append(w * h)
        avg_area = sum(areas) / len(areas) if areas else 0.0

        target = {
            'storey': storey_cls,  # int64
            'room_tot': torch.tensor(room_tot, dtype=torch.float32),
            'room_per': room_per,  # (9,)
            'avg_area': torch.tensor(avg_area, dtype=torch.float32),
        }

        return torch.from_numpy(pc), target
