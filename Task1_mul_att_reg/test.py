# test_multi.py
import os, torch, csv
from utils import load_step, sample_shape, normalise
from model import PointNetMulti
import random

device  = 'cuda' if torch.cuda.is_available() else 'cpu'
net     = PointNetMulti().to(device)
net.load_state_dict(torch.load('pointnet_multi.pth', map_location=device))
net.eval()

out_path = 'predict_multi.txt'
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['num', 'pred_storey', 'pred_room_tot',
                     'pred_room_per_F1-F10', 'pred_avg_area'])

    tempaa=os.listdir('data_total')
    #随机选取100个进行测试
    tempaa=random.sample(tempaa, 100)
    for file in sorted(tempaa):
        bid = file.split('.')[0]

        # --- 点云准备 (与训练同规) ------------------------
        pc = sample_shape(load_step(f'data/{file}'), 2048)
        pc = normalise(pc)[None]                       # (1,2048,3)
        pc = torch.from_numpy(pc).to(device)

        # --- 推理 ---------------------------------------
        with torch.no_grad():
            out = net(pc)
        storey    = out['storey_logits'].argmax(1).item() + 2
        room_tot  = int(out['room_tot'].round().item())
        room_per  = out['room_per'].round().int().cpu().tolist()[0]   # list 长 10
        avg_area  = float(out['avg_area'].item())

        writer.writerow([bid, storey, room_tot, room_per, f'{avg_area:.2f}'])
print('✅ results written to', out_path)
