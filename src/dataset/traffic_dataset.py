import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from torch_geometric.data import Data
import numpy as np
from .dataset_config import (
    edge_idx_map,
    node_idx_map,
    edge_adj_mat,
    edge_degree_list,
    edge_spd,
    edge_index,
    edge_attr
)

# DataLoader 배치 시 edge_index/attr 공유되도록 Long/FloatTensor 변환
EDGE_INDEX = torch.from_numpy(edge_index).long()    # shape [2, E]
EDGE_ATTR  = torch.from_numpy(edge_attr).float()    # shape [E, F_e]


class TrafficDataset(Dataset):
    """
    PyG Data 객체를 반환하도록 수정한 TrafficDataset.
    """
    def __init__(self, traffic_data, window=12, week_steps=480*7):
        super().__init__()
        self.traffic   = traffic_data          # (T_total, E, C_all) as NumPy
        self.window    = window
        self.week_steps = week_steps
        self.day_steps  = week_steps // 7
        self.E         = traffic_data.shape[1]

        T_total = traffic_data.shape[0]
        self.min_start = (T_total // week_steps - 1) * week_steps + (self.window - 1)
        self.max_start = T_total - 12 - 1
        self.starts    = list(range(self.min_start, self.max_start + 1))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        t0 = self.starts[idx]

        # 과거/미래 인덱스
        past_idxs   = np.arange(t0 - self.window + 1, t0 + 1)
        fut_offsets = np.array([3, 6, 12], dtype=np.int64)
        fut_idxs    = t0 + fut_offsets

        # raw
        past = self.traffic[past_idxs]   # (12, E, C_all)
        fut  = self.traffic[fut_idxs]    # (3,  E, C_all)

        # 채널 0~2: volume, density, flow
        Xp = torch.from_numpy(past[..., :3]).float()   # (12, E, 3)
        Xf = torch.from_numpy(fut[...,  :3]).float()   # (3,  E, 3)

        # 시간 특성
        tod_enc = ((past_idxs % self.day_steps) * 24.0 / self.day_steps).astype(np.float32)
        dow_enc = ((past_idxs // self.day_steps) % 7).astype(np.int64)
        tod_dec = ((fut_idxs % self.day_steps) * 24.0 / self.day_steps).astype(np.float32)
        dow_dec = ((fut_idxs // self.day_steps) % 7).astype(np.int64)

        # (seq, E, 1)
        tod_feat_enc = torch.from_numpy(tod_enc)[:, None, None].expand(-1, self.E, 1)
        dow_feat_enc = torch.from_numpy(dow_enc).float()[:, None, None].expand(-1, self.E, 1)
        tod_feat_dec = torch.from_numpy(tod_dec)[:, None, None].expand(-1, self.E, 1)
        dow_feat_dec = torch.from_numpy(dow_dec).float()[:, None, None].expand(-1, self.E, 1)

        # past_edges: (T, E, D_in)
        x = torch.cat([Xp, tod_feat_enc, dow_feat_enc], dim=-1)  # (12, E, 5)

        # future_edges → y: flatten to (n_pred * E, D_tar)
        future_edges = torch.cat([Xf, tod_feat_dec, dow_feat_dec], dim=-1)[..., :3]  # (3, E, 3)
        n_pred, E, D_tar = future_edges.shape
        y = future_edges.reshape(n_pred * E, D_tar)  # (n_pred*E, 3)

        # PyG Data 객체 생성
        data = Data(
            x=x,                        # [T, E, D_in]
            edge_index=EDGE_INDEX,      # [2, E]
            edge_attr=EDGE_ATTR,        # [E, F_e]
            y=y                         # [n_pred*E, D_tar]
        )

        return data
