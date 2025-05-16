# inference_classify.py  ── Good / Defect 推理
import sys, torch
from pathlib import Path
from utils import load_step, sample_shape, normalise

# ① 确保与训练时同名同结构的模型；若训练用 PointNetSeg，请对应导入
from model import PointNet    # 训练时就是 PointNet(num_classes=2)

def is_defect(step_file: str,
              model_file: str = 'pointnet_defect.pth',
              n_pts: int = 4000,
              strict: bool = False) -> bool:
    """
    返回 True 表示 Defect，False 表示 Good。
    strict=False 可在层名不完全一致时容忍缺失 / 冗余键。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) 建立网络
    net = PointNet(num_classes=2).to(device)

    # 2) 读取 checkpoint ─ 既能兼容 “纯 state_dict”，也能兼容
    #    {"epoch":..., "model_state":..., "opt_state":...} 这种完整 checkpoint
    ckpt = torch.load(model_file, map_location=device)
    state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt

    # 如果训练时用了 nn.DataParallel，键会带 "module." 前缀，去掉：
    state = {k.replace('module.', ''): v for k, v in state.items()}

    # 3) 载入权重
    net.load_state_dict(state, strict=strict)
    net.eval()

    # 4) STEP → 点云 → 归一化
    pts = normalise(sample_shape(load_step(step_file), n_pts))
    pc  = torch.from_numpy(pts[None].astype('float32')).to(device)

    # 5) 推理
    with torch.no_grad():
        cls = net(pc).argmax(1).item()   # 0:Good, 1:Defect
    return cls == 1


# ------------------------------------------------------------------
if __name__ == '__main__':
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: python inference_classify.py file.step [model.pth]")
        sys.exit(1)

    step_path = sys.argv[1]
    model_pth = sys.argv[2] if len(sys.argv) == 3 else 'pointnet_defect.pth'

    defect = is_defect(step_path, model_file=model_pth)
    label  = 'DEFECT' if defect else 'GOOD'
    icon   = '🛑' if defect else '✅'
    print(f"{icon} {Path(step_path).name} → {label}")
