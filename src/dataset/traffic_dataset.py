import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import random
from .dataset_config import (
    edge_index,
    edge_attr,
    week_steps
)

# 전역으로 한 번만 변환
EDGE_INDEX = torch.from_numpy(edge_index).long()
EDGE_ATTR  = torch.from_numpy(edge_attr).float()

class TrafficDataset(Dataset):
    """
    PyG Data 객체를 반환.
    randomize=False: 기존처럼 순서대로,
    randomize=True: __getitem__마다 self.starts에서 랜덤 샘플링.
    """
    def __init__(self, traffic_data, window=12, week_steps=week_steps, randomize=False):
        super().__init__()
        self.traffic    = traffic_data      # (T_total, E, C_all) as NumPy
        self.window     = window
        self.week_steps = week_steps
        self.day_steps  = week_steps // 7
        self.E          = traffic_data.shape[1]
        self.randomize  = randomize

        T_total = traffic_data.shape[0]
        self.min_start = (T_total // week_steps - 1) * week_steps + (self.window - 1)
        self.max_start = T_total - 12 - 1
        self.starts    = list(range(self.min_start, self.max_start + 1))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        # 1) t0 결정
        if self.randomize:
            t0 = random.choice(self.starts)
        else:
            t0 = self.starts[idx]

        # 2) 과거/미래 인덱스
        past_idxs   = np.arange(t0 - self.window + 1, t0 + 1)
        fut_offsets = np.array([3, 6, 12], dtype=np.int64)
        fut_idxs    = t0 + fut_offsets

        # 3) raw slice
        past = self.traffic[past_idxs]   # (T, E, C_all)
        fut  = self.traffic[fut_idxs]    # (n_pred, E, C_all)

        # 4) 채널 0~2: volume, density, flow
        Xp = torch.from_numpy(past[..., :3]).float()  # (T, E, 3)
        Xf = torch.from_numpy(fut[...,  :3]).float()  # (n_pred, E, 3)

        # 5) 시간 특성
        tod_enc = ((past_idxs % self.day_steps) * 24.0 / self.day_steps).astype(np.float32)
        dow_enc = ((past_idxs // self.day_steps) % 7).astype(np.int64)
        tod_dec = ((fut_idxs   % self.day_steps) * 24.0 / self.day_steps).astype(np.float32)
        dow_dec = ((fut_idxs   // self.day_steps) % 7).astype(np.int64)

        tod_feat_enc = torch.from_numpy(tod_enc)[:, None, None].expand(-1, self.E, 1)
        dow_feat_enc = torch.from_numpy(dow_enc).float()[:, None, None].expand(-1, self.E, 1)
        tod_feat_dec = torch.from_numpy(tod_dec)[:, None, None].expand(-1, self.E, 1)
        dow_feat_dec = torch.from_numpy(dow_dec).float()[:, None, None].expand(-1, self.E, 1)

        # 6) 입력 x, 목표 y 구성
        x = torch.cat([Xp, tod_feat_enc, dow_feat_enc], dim=-1)      # [T, E, D_in]
        future_edges = torch.cat([Xf, tod_feat_dec, dow_feat_dec], dim=-1)[..., :3]
        y = future_edges                                            # [n_pred, E, D_out]

        # 7) Data 객체 반환
        return Data(
            x=x,                        # [T, E, D_in]
            edge_index=EDGE_INDEX,      # [2, E]
            edge_attr=EDGE_ATTR,        # [E, F_e]
            y=y                         # [n_pred, E, D_out]
        )
