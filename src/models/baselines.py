# src/modules/baseline.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, MessagePassing

__all__ = ['GCNMLP', 'DCRNN', 'STGCN']

#
# 1) GCN + MLP 분류/회귀 모델
#
class GCNMLP(nn.Module):
    def __init__(
        self,
        num_nodes: int,            # E
        node_feature_dim: int,     # D_in
        pred_node_dim: int,        # D_out
        n_pred: int = 1,           # 출력 스텝 수
        encoder_embed_dim: int = 128,
        encoder_depth: int = 2,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 64,
        mlp_pred_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_pred    = n_pred
        self.embed = nn.Linear(node_feature_dim, encoder_embed_dim)
        self.convs = nn.ModuleList([
            ChebConv(encoder_embed_dim, encoder_embed_dim, K=2)
            for _ in range(encoder_depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_pred_dropout),
            nn.Linear(mlp_hidden_dim, pred_node_dim),
        )

    def forward(self, x, edge_index, edge_attr=None):
        # x: [B, T, E, D_in] or [B, E, D_in]
        if x.dim() == 4:
            x = x.mean(dim=1)               # [B, E, D_in]
        B, E, D = x.shape
        h = self.embed(x)                  # [B, E, embed]
        h = h.view(B*E, -1)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        h = h.view(B, E, -1).mean(dim=1)   # [B, embed]
        out1 = self.mlp(h)                 # [B, pred_node_dim]
        # multi-step 복제
        out = out1.unsqueeze(1).repeat(1, self.n_pred, 1)  # [B, n_pred, pred_node_dim]
        # 엣지 차원 삽입: [B, n_pred, E, D_out]
        return out.unsqueeze(2).expand(-1, -1, E, -1)

#
# 2) DCRNN: Diffusion Conv + GRU 스타일 RNN
#
class DConv(MessagePassing):
    def __init__(self, in_c, out_c, K, bias=True):
        super().__init__(aggr="add")
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(K, in_c, out_c))
        self.bias   = nn.Parameter(torch.Tensor(out_c)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        B, N, F = x.shape
        out = torch.zeros(B, N, self.weight.size(2), device=x.device)
        H = x
        for k in range(self.K):
            H = self.propagate(edge_index, x=H, norm=edge_weight)
            out += H @ self.weight[k]
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j, norm):
        return norm.unsqueeze(-1) * x_j

class DCRNNLayer(nn.Module):
    def __init__(self, in_c, out_c, K):
        super().__init__()
        self.dconv = DConv(in_c+out_c, out_c, K)

    def forward(self, x, h, edge_index, edge_weight):
        z = torch.sigmoid(self.dconv(torch.cat([x, h], dim=-1), edge_index, edge_weight))
        r = torch.sigmoid(self.dconv(torch.cat([x, h], dim=-1), edge_index, edge_weight))
        h_tilde = torch.tanh(self.dconv(torch.cat([x, r*h], dim=-1), edge_index, edge_weight))
        return z * h + (1-z) * h_tilde

class DCRNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_feature_dim: int,
        pred_node_dim: int,
        n_pred: int = 1,
        encoder_embed_dim: int = 64,
        encoder_depth: int = 2,
        K: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_pred    = n_pred
        self.layers = nn.ModuleList([
            DCRNNLayer(
                node_feature_dim if i==0 else encoder_embed_dim,
                encoder_embed_dim, K
            )
            for i in range(encoder_depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(encoder_embed_dim, pred_node_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # x: [B, T, N, D_in]
        # edge_attr: [E, F_e] (거리 + 타입원핫)
        # 우리는 edge_weight로 스칼라 거리만 사용할 것
        if edge_attr is not None and edge_attr.dim() == 2:
            edge_weight = edge_attr[:, 0]        # 거리 값만
        else:
            edge_weight = edge_attr              # 이미 스칼라인 경우

        B, T, N, D = x.shape
        h = torch.zeros(B, N, self.layers[0].dconv.weight.size(2), device=x.device)
        for t in range(T):
            x_t = x[:, t]
            for layer in self.layers:
                h = layer(x_t, h, edge_index, edge_weight)
                h = F.relu(h)
                h = self.dropout(h)
        # single-step 예측 [B, N, pred_dim]
        out1 = self.pred(h)
        # multi-step 복제 → [B, n_pred, N, pred_dim]
        return out1.unsqueeze(1).repeat(1, self.n_pred, 1, 1)

#
# 3) STGCN: 시공간 합성곱 블록
#
class TemporalConv(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, (1, k))
        self.conv2 = nn.Conv2d(in_c, out_c, (1, k))
        self.conv3 = nn.Conv2d(in_c, out_c, (1, k))

    def forward(self, x):
        # x: [B, T, N, F] → [B, F, N, T]
        x = x.permute(0,3,2,1)
        p = self.conv1(x)
        q = torch.sigmoid(self.conv2(x))
        h = p * q + self.conv3(x)
        h = F.relu(h)
        return h.permute(0,3,2,1)

class STGCNBlock(nn.Module):
    def __init__(self, in_c, hidden_c, out_c, k_sz, K, norm='sym'):
        super().__init__()
        self.temp1 = TemporalConv(in_c, hidden_c, k_sz)
        self.gconv = ChebConv(hidden_c, hidden_c, K, normalization=norm)
        self.temp2 = TemporalConv(hidden_c, out_c, k_sz)
        self.bn    = nn.BatchNorm2d(hidden_c)

    def forward(self, x, edge_index, edge_attr=None):
        # x: [B, T, N, F]
        t1 = self.temp1(x)   # [B, T', N, hidden_c]
        B, T1, N, C = t1.shape

        # edge_attr: [E, F_e] → 스칼라 가중치(거리)만 사용
        if edge_attr is not None and edge_attr.dim() == 2:
            edge_weight = edge_attr[:, 0]   # 첫 번째 채널(거리)
        else:
            edge_weight = edge_attr         # 이미 1D일 때

        out = torch.zeros(B, T1, N, C, device=x.device)
        for b in range(B):
            for t in range(T1):
                out[b, t] = self.gconv(t1[b, t], edge_index, edge_weight)

        out = F.relu(out)
        out = self.temp2(out)
        # batch norm expects shape [B, C, N, T']
        out_bn = out.permute(0,3,2,1)      # [B, hidden_c, N, T']
        out_bn = self.bn(out_bn)
        return out_bn.permute(0,3,2,1)    # [B, T', N, hidden_c]

class STGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_feature_dim: int,
        pred_node_dim: int,
        n_pred: int = 1,
        encoder_embed_dim: int = 64,
        encoder_depth: int = 2,
        kernel_size: int = 3,
        K: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_pred    = n_pred
        self.blocks = nn.ModuleList([
            STGCNBlock(
                node_feature_dim if i==0 else encoder_embed_dim,
                encoder_embed_dim,
                encoder_embed_dim,
                kernel_size,
                K
            )
            for i in range(encoder_depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(encoder_embed_dim, pred_node_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # x: [B, T, N, D_in]
        h = x
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
            h = self.dropout(h)
        # h: [B, T_final, N, embed]
        h_mean = h.mean(dim=1)            # [B, N, embed]
        out1 = self.pred(h_mean)          # [B, N, D_out]
        # multi-step 복제 → [B, n_pred, N, D_out]
        return out1.unsqueeze(1).repeat(1, self.n_pred, 1, 1)
