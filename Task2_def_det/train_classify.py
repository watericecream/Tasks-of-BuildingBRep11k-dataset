# train_classify.py — PointNet 二分类训练 + Resume + LR Scheduler + Checkpoint
import argparse
import time
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataset_classify import DefectDataset
from model import PointNet  # PointNet(num_classes=2)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # — 数据加载 —
    train_ds = DefectDataset(split='train', seed=42)
    val_ds   = DefectDataset(split='val',   seed=42)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)

    # — 模型 / 优化器 / 损失 & 度量 / 调度 —
    model      = PointNet(num_classes=2).to(device)
    opt        = torch.optim.Adam(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler  = StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion  = torch.nn.CrossEntropyLoss()
    acc_metric = MulticlassAccuracy(num_classes=2).to(device)

    # — 日志与 Checkpoint 目录 —
    log_path = Path('log_classify.csv')
    ckpt_dir = Path('checkpoints_classify'); ckpt_dir.mkdir(exist_ok=True)
    # 写入表头
    with log_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','val_acc','lr'])

    # — Resume if requested —
    start_epoch = 0
    if args.resume:
        print(f"🔄 Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # 如果 checkpoint 中保存了字典
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
            opt.load_state_dict(ckpt['opt_state'])
            start_epoch = ckpt.get('epoch', 0)
        else:
            model.load_state_dict(ckpt)
        print(f"   Resumed at epoch {start_epoch}")

    # — 训练循环 —
    for epoch in range(start_epoch, args.epochs):
        # — Train —
        model.train()
        tot_loss = 0.0
        t0 = time.time()
        for pts, y in tqdm.tqdm(train_dl, desc=f'E{epoch:02d}', leave=False):
            pts = pts.to(device).float()
            y   = y.to(device)

            logits = model(pts)
            loss   = criterion(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss += loss.item() * pts.size(0)

        train_loss = tot_loss / len(train_ds)
        elapsed = time.time() - t0

        # — Validate —
        model.eval()
        acc_metric.reset()
        with torch.no_grad():
            for pts, y in val_dl:
                pts = pts.to(device).float()
                y   = y.to(device)
                logits = model(pts)
                acc_metric.update(logits, y)
        val_acc = acc_metric.compute().item()

        # — Current learning rate —
        lr = scheduler.get_last_lr()[0]

        # — Logs & Checkpoint —
        print(f'E{epoch:02d} train_loss {train_loss:.4f}  val_acc {val_acc:.4f}  lr={lr:.1e}  [{elapsed:.1f}s]')
        with log_path.open('a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{train_loss:.4f}', f'{val_acc:.4f}', f'{lr:.1e}'])

        # 保存 checkpoint（含 epoch、模型权重、优化器状态）
        torch.save({
            'epoch'      : epoch + 1,
            'model_state': model.state_dict(),
            'opt_state'  : opt.state_dict()
        }, ckpt_dir / f'epoch{epoch:02d}.pth')

        # 学习率衰减
        scheduler.step()

    # — 保存最终模型 —
    final_pth = 'pointnet_defect_final.pth'
    torch.save(model.state_dict(), final_pth)
    print(f"✅ Training finished. Final model → {final_pth}")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="PointNet Defect Classification Training")
    parser.add_argument('--resume',    type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs',    type=int,   default=200,
                        help='Total epochs to train (default: 200)')
    parser.add_argument('--batch-size',type=int,   default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr',        type=float, default=1e-3,
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--lr-step',   type=int,   default=10,
                        help='LR scheduler step size (default: 10)')
    parser.add_argument('--lr-gamma',  type=float, default=0.1,
                        help='LR scheduler decay factor (default: 0.1)')
    args = parser.parse_args()
    main(args)
