# src/modules/STGCN_with_auxiliary_network_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.baselines import STGCN

class NodeTrendEncoder(nn.Module):
    """노드별 과거 12 step 을 Conv1d 로 요약 → (E, H)"""
    def __init__(self, C_in: int, H: int, kernel_size: int = 3, T_win: int = 12):
        super().__init__()
        self.conv = nn.Conv1d(C_in, H, kernel_size, padding=kernel_size // 2)
        self.bn   = nn.BatchNorm1d(H)
        self.T_win, self.H = T_win, H

    def forward(self, x_win: torch.Tensor) -> torch.Tensor:
        # x_win: [E, C_in, T_win]
        h = F.relu(self.bn(self.conv(x_win)))   # [E, H, T_win]
        return h.mean(dim=2)                    # [E, H]

class STGCNWithAux(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 node_feature_dim: int,   # C_in = 5
                 pred_node_dim: int,
                 aux_data: torch.Tensor,
                 n_pred: int = 3,
                 encoder_embed_dim: int = 32,
                 aux_embed_dim: int = 32,
                 encoder_depth: int = 1,
                 ):
        super().__init__()
        self.aux_H = aux_embed_dim
        # ----- 보조 네트워크 -----
        self.trend_enc = NodeTrendEncoder(
            C_in=3,          # volume/density/flow
            H=aux_embed_dim
        )
        # ----- 1×1 Conv Projection -----
        self.proj = nn.Conv2d(
            in_channels=node_feature_dim - 2 + aux_embed_dim,  # 3+H
            out_channels=node_feature_dim - 2,                 # 3
            kernel_size=1
        )
        # ----- 메인 모델(STGCN) -----
        self.main = STGCN(
            num_nodes=num_nodes,
            node_feature_dim=node_feature_dim - 2,   # 3 채널만 전달
            pred_node_dim=pred_node_dim,
            n_pred=n_pred,
            encoder_embed_dim=encoder_embed_dim,
            encoder_depth=encoder_depth
        )
        # 보조 데이터(3주치) 로드 → register_buffer
        self.register_buffer("aux_data", aux_data)

        from dataset.dataset_config import week_steps, day_steps
        self.W = week_steps
        self.D = day_steps

    @torch.no_grad()
    def _get_slot_ids(self, dow: torch.Tensor, tod: torch.Tensor):
        # dow,tod: [B,T,E]   (tod 는 0~24 부동소수)
        tod_step = torch.round(tod / 24 * self.D).long()        # 0~479
        slot_id  = dow.long() * self.D + tod_step               # 0~3359
        return slot_id  # [B,T,E]

    def _query_aux(self, slot_id):
        """
        slot_id: [B, T_in, E]  (int64)
        returns: aux_emb [B, E, H]
        """
        B, T_in, E = slot_id.shape

        # 1) 기준 시점만: slot_id[:,0] → s0
        s0 = slot_id[:, 0, :]  # [B, E]

        # 2) 3주치 time idx 생성 (no nonzero)
        offs = torch.tensor([0, self.W, 2*self.W], device=s0.device)
        idxs = (s0.unsqueeze(-1) + offs.unsqueeze(0)) % self.aux_data.size(0)  # [B, E, 3]

        # 3) aux_data: [T_aux, E, C] → [E, C, T_aux]
        aux = self.aux_data.permute(1,2,0)  # [E, C, T_aux]
        # expand once: [B, E, C, T_aux]
        aux = aux.unsqueeze(0).expand(B, -1, -1, -1)

        # 4) gather로 한 번에 뽑기 (no aten::index)
        # idxs: [B,E,3] → expand to [B,E,C,3]
        idxs_exp = idxs.unsqueeze(2).expand(-1,-1,aux.size(2),-1)
        slice_ = aux.gather(dim=3, index=idxs_exp)  # [B, E, C, 3]

        # 5) batch화된 trend_enc
        BE, C = B*E, slice_.size(2)
        # reshape [B*E, C, 3]
        X = slice_.reshape(BE, C, 3)
        Z = self.trend_enc(X)         # [B*E, H]
        return Z.view(B, E, -1)       # [B, E, H]


    def forward(self, x, edge_index, edge_attr):
        # x: [B, T_in, E, 5] (vol,dens,flow,tod,dow)
        vol_flow = x[..., :3]             # [B,T,E,3]
        tod = x[..., 3];  dow = x[..., 4]
        slot_id = self._get_slot_ids(dow, tod)  # [B,T,E]

        # ----- 보조 임베딩 생성 -----
        aux_emb = self._query_aux(slot_id)      # [B,E,H]
        aux_exp = aux_emb.unsqueeze(1).expand(-1, vol_flow.size(1), -1, -1)
                                               # [B,T,E,H]

        # ----- concat + 1×1 conv 투영 -----
        x_cat = torch.cat([vol_flow, aux_exp], dim=-1)   # [B,T,E,3+H]
        # 1×1 conv expects (B,C,H,W) → (B,feat,E,T)
        x_cat = x_cat.permute(0, 3, 2, 1)                # [B,3+H,E,T]
        x_proj = self.proj(x_cat).permute(0, 3, 2, 1)    # [B,T,E,3]

        # ----- 메인 STGCN -----
        return self.main(x_proj, edge_index, edge_attr)
