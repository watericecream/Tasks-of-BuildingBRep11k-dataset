# model.py  ── PointNet encoder + 多任务头
import torch, torch.nn as nn, torch.nn.functional as F

# ----------------------------------------------------------------------
class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.conv1 = nn.Conv1d(k,   64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128,1024, 1)
        self.fc1, self.fc2, self.fc3 = nn.Linear(1024,512), nn.Linear(512,256), nn.Linear(256,k*k)
        self.bn1 = nn.BatchNorm1d(64);  self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
        nn.init.zeros_(self.fc3.weight); nn.init.eye_(self.fc3.bias.view(k,k))

    def forward(self, x):
        B,k,N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))).max(-1)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        return self.fc3(x).view(B,k,k)

# ----------------------------------------------------------------------
class PointNetEncoder(nn.Module):
    """Shared backbone that outputs a 1024-d global feature."""
    def __init__(self):
        super().__init__()
        self.tnet  = TNet(3)
        self.conv1 = nn.Conv1d(3,   64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128,1024, 1)
        self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):               # x: (B,N,3)
        B,N,_ = x.size()
        x = x.permute(0,2,1)            # (B,3,N)
        x = torch.bmm(self.tnet(x), x)  # alignment
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))).max(-1)[0]   # (B,1024)
        return x

# ----------------------------------------------------------------------
class PointNetMulti(nn.Module):
    """Multi-task head: storey cls + total rooms + per-floor rooms + avg area"""
    def __init__(self):
        super().__init__()
        self.backbone      = PointNetEncoder()     # (B,1024)
        self.head_storey   = nn.Linear(1024, 9)    # 分类：9
        self.head_roomtot  = nn.Linear(1024, 1)    # 回归：1
        self.head_roomper  = nn.Linear(1024, 10)    # 回归：10
        self.head_avgarea  = nn.Linear(1024, 1)    # 回归：1

    def forward(self, x):                          # x: (B,N,3)
        feat = self.backbone(x)                    # (B,1024)
        return {
            'storey_logits': self.head_storey(feat),          # (B,9)
            'room_tot':      self.head_roomtot(feat).squeeze(-1),   # (B,)
            'room_per':      self.head_roomper(feat),               # (B,9)
            'avg_area':      self.head_avgarea(feat).squeeze(-1),   # (B,)
        }

# ----------------------------------------------------------------------
# 仍保留最初的单任务分类版 PointNet，如要继续用可直接 import PointNet
class PointNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.fc1, self.fc2 = nn.Linear(1024,512), nn.Linear(512,256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512); self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        feat = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)
