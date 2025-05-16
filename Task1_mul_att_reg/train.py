# train.py — PointNetMulti 多任务训练 + Resume + LR Scheduler + Checkpoint
import argparse
import time
import math
import csv
import pathlib

import torch
import torch.nn.functional as F
import tqdm
from torchmetrics.classification import MulticlassAccuracy

from dataset import BuildingDataset
from model import PointNetMulti

def main(args):
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据加载
    train_ds = BuildingDataset('train')
    val_ds   = BuildingDataset('val')
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_dl   = torch.utils.data.DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    # 模型、优化器、调度器、指标
    model   = PointNetMulti().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    acc_cls = MulticlassAccuracy(num_classes=9).to(device)
    crit_cls= torch.nn.CrossEntropyLoss()

    # 日志 & checkpoint 目录
    log_path = pathlib.Path('log_multi.csv')
    ckpt_dir = pathlib.Path('checkpoints'); ckpt_dir.mkdir(exist_ok=True)
    with log_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','val_acc','rmse_roomtot','mae_avgarea','lr'])

    # 如果指定了 --resume，就加载已有 checkpoint
    start_epoch = 0
    if args.resume:
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # 如果只保存了 state_dict：
        if 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
            opt.load_state_dict(ckpt['opt_state'])
            start_epoch = ckpt.get('epoch', 0)
        else:
            model.load_state_dict(ckpt)
        print(f"   Resumed at epoch {start_epoch}")

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tot_loss = 0.0
        t0 = time.time()
        for pc, tgt in tqdm.tqdm(train_dl, desc=f'E{epoch:02d}', leave=False):
            pc            = pc.to(device, non_blocking=True)
            tgt_storey    = tgt['storey'].to(device, non_blocking=True)
            tgt_room_tot  = tgt['room_tot'].to(device)
            tgt_room_per  = tgt['room_per'].to(device)
            tgt_avg_area  = tgt['avg_area'].to(device)

            out = model(pc)
            loss  = crit_cls(out['storey_logits'], tgt_storey)
            loss += F.l1_loss(out['room_tot'], tgt_room_tot)
            loss += F.mse_loss(out['room_per'], tgt_room_per)
            loss += F.mse_loss(out['avg_area'], tgt_avg_area)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss += loss.item() * pc.size(0)

        train_loss = tot_loss / len(train_ds)
        elapsed = time.time() - t0
        print(f'E{epoch:02d} train-loss {train_loss:.3f} [{elapsed:.1f}s]')

        # 验证
        model.eval()
        acc_cls.reset()
        se_rt = 0.0
        ae_aa = 0.0
        with torch.no_grad():
            for pc, tgt in val_dl:
                pc = pc.to(device, non_blocking=True)
                out= model(pc)
                acc_cls.update(out['storey_logits'],
                               tgt['storey'].to(device))
                se_rt += ((out['room_tot'].cpu() - tgt['room_tot'])**2).sum().item()
                ae_aa += (out['avg_area'].cpu() - tgt['avg_area']).abs().sum().item()

        val_acc      = acc_cls.compute().item()
        rmse_roomtot = math.sqrt(se_rt/len(val_ds))
        mae_avgarea  = ae_aa/len(val_ds)
        print(f'           val acc {val_acc:.3f} | '
              f'rmse_roomtot {rmse_roomtot:.2f} | mae_avgarea {mae_avgarea:.2f}')

        # 记录当前学习率
        lr = scheduler.get_last_lr()[0]

        # 写日志
        with log_path.open('a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f'{train_loss:.4f}', f'{val_acc:.4f}',
                f'{rmse_roomtot:.2f}', f'{mae_avgarea:.2f}', f'{lr:.1e}'
            ])

        # 保存 checkpoint（含 epoch, model_state, opt_state）
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'opt_state':   opt.state_dict()
        }, ckpt_dir / f'epoch{epoch:02d}.pth')

        # 学习率衰减
        scheduler.step()

    # 保存最终模型
    torch.save(model.state_dict(), 'pointnet_multi.pth')
    print('✅ Training finished; final model → pointnet_multi_final.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path to resume from')
    parser.add_argument('--epochs', type=int, default=50,
                        help='total number of epochs to train')
    args = parser.parse_args()
    main(args)
