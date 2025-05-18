# src/modules/baseline.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, MessagePassing

__all__ = ['GCNMLP', 'DCRNN', 'STGCN', 'MLPBASED']

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
        # time-dim 패딩: (height_pad, width_pad)
        # height_pad=0 (node 차원에는 패딩 없음),
        # width_pad=(k-1)//2 로 중앙정렬 패딩
        pad = (0, (k-1)//2)
        self.conv1 = nn.Conv2d(in_c, out_c, (1, k), padding=pad)
        self.conv2 = nn.Conv2d(in_c, out_c, (1, k), padding=pad)
        self.conv3 = nn.Conv2d(in_c, out_c, (1, k), padding=pad)

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


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, edge_index, edge_attr=None):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]
        

        if self.tod_embedding_dim > 0:
            tod = x[..., 3]/self.steps_per_day
        
        if self.dow_embedding_dim > 0:
            dow = x[..., 4]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
            

        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out

class MLPBASED(nn.Module):
    def __init__(self, T, E, D_in, n_pred, D_out, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.n_pred, self.E, self.D_out = n_pred, E, D_out
        self.network = nn.Sequential(
            nn.Flatten(),  # B x (T*E*D_in)
            nn.Linear(T * E * D_in, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_pred * E * D_out)
        )

    def forward(self, x, edge_index=None, edge_attr=None):
        # x: [B, T, E, D_in]
        B = x.size(0)
        out = self.network(x)
        out = out.view(B, self.n_pred, self.E, self.D_out)
        return out


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, *, dim]
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out + residual


class ResidualMLPBaseline(nn.Module):
    def __init__(self, T, E, D_in, n_pred, D_out, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.n_pred, self.E, self.D_out = n_pred, E, D_out
        self.input_dim = T * E * D_in
        self.hidden_dim = hidden_dim

        # 입력을 hidden_dim 차원으로 투사
        self.fc_in = nn.Linear(self.input_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual 블록 3개
        self.res_blocks = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout),
            ResidualMLPBlock(hidden_dim, dropout),
            ResidualMLPBlock(hidden_dim, dropout),
        )

        # 출력층
        self.fc_out = nn.Linear(hidden_dim, n_pred * E * D_out)

    def forward(self, x, edge_index=None, edge_attr=None):
        # x: [B, T, E, D_in]
        B = x.size(0)
        # Flatten 및 프로젝션
        h = x.view(B, -1)            # [B, T*E*D_in]
        h = self.fc_in(h)            # [B, hidden_dim]
        h = self.ln_in(h)
        h = F.relu(h)
        h = self.dropout(h)

        # Residual MLP 블록
        h = self.res_blocks(h)       # [B, hidden_dim]

        # 예측치 생성
        out = self.fc_out(h)         # [B, n_pred*E*D_out]
        out = out.view(B, self.n_pred, self.E, self.D_out)
        return out