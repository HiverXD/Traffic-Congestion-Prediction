import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F
#from torch_scatter import scatter_add


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearTemporal(nn.Module):
    def __init__(self, in_steps, out_steps, num_nodes, kernel_size, individual=True):
        super().__init__()
        self.decomp = series_decomp(kernel_size)
        self.in_steps, self.out_steps = in_steps, out_steps
        self.num_nodes = num_nodes
        self.individual = individual

        if individual:
            self.lin_season = nn.ModuleList([
                nn.Linear(in_steps, out_steps) for _ in range(num_nodes)
            ])
            self.lin_trend = nn.ModuleList([
                nn.Linear(in_steps, out_steps) for _ in range(num_nodes)
            ])
        else:
            self.lin_season = nn.Linear(in_steps, out_steps)
            self.lin_trend  = nn.Linear(in_steps, out_steps)

    def forward(self, x, dim=1):
        # x: [B, in_steps, num_nodes, model_dim]
        B, T, N, D = x.shape
        # first collapse model_dim into channel (we do DLinear on each feature separately)
        x = x.permute(0, 2, 3, 1).reshape(B*N*D, T)  # [B·N·D, T]
        res, mean = self.decomp(x.unsqueeze(-1))  # decomp expects shape [B·N·D, T, 1]
        res = res.squeeze(-1); mean = mean.squeeze(-1)

        # apply linear
        if self.individual:
            # but now each node & each feature has its own linear?
            # simplest is to ignore feature-dim and just apply same for each D
            out_res = torch.stack([
                self.lin_season[n](res[B*n*D:(B*(n+1)*D), :])  for n in range(N)
            ], dim=1)  # [B, N, D, out_steps]
            out_mean = torch.stack([
                self.lin_trend[n](mean[B*n*D:(B*(n+1)*D), :])   for n in range(N)
            ], dim=1)
        else:
            out_res  = self.lin_season(res)  # [B·N·D, out_steps]
            out_mean = self.lin_trend(mean)

        out = out_res + out_mean
        out = out.view(B, N, D, self.out_steps).permute(0, 3, 1, 2)
        # → [B, out_steps, num_nodes, model_dim]
        return out

# ---------- 1) RRWP Positional Encoding ----------
def rrwp_encoding(edge_index, num_nodes, K=3):
    """
    edge_index : [2, E]  (COO)
    return      : tensor [E, K+1]  stacked RRWP features for each edge (i->j)
    """
    # ① 인접행렬 스파스 표현
    row, col = edge_index
    deg = scatter_add(torch.ones_like(row, dtype=torch.float), row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg.clamp(min=1)
    norm = deg_inv[row]                 # D^{-1}
    
    # ② 1-step 확률 M : (i→j) = 1/deg(i)
    P1 = norm                           # shape [E]
    
    # ③ 다중 홉: P^k = M^k  (소규모 그래프 기준 간단 반복)
    P_list = [torch.ones_like(P1)]      # k=0 (I)
    P_k = P1.clone()
    for _ in range(K):
        P_list.append(P_k)
        # 다음 홉 확률 계산: (M @ M^{k})_ij
        #   여기서는 간단화를 위해 edge-list 집계로 근사 (소규모 그래프용)
        P_k = scatter_add(P_k, row, dim=0, dim_size=num_nodes)[row] * P1
    
    return torch.stack(P_list, dim=-1)  # [E, K+1]


# ---------- 2) RRWP-MLP 변환 ----------
class RRWPEmbedding(nn.Module):
    def __init__(self, K, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(K+1, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    def forward(self, rrwp):
        return self.mlp(rrwp)           # [E, d']


class GRITAttention(nn.Module):
    def __init__(self, d_model, d_edge, n_heads):
        super().__init__()
        self.d_h = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_Ew = nn.Linear(d_edge, d_model)
        self.W_Ev = nn.Linear(d_edge, d_model)
        self.W_A  = nn.Linear(d_model, 1, bias=False)
        self.n_heads = n_heads

    def forward(self, x, edge_index, e_emb, batch_index):
        """
        x        : [B, N, d_model] (batch or single)
        edge_index : [2, E] (torch.long)
        e_emb      : [E, d_edge]
        batch_index : [E] (각 edge의 번호)
        """
        B, N, d = x.shape
        E = edge_index.shape[1]
        # Q, K, V 생성
        Q = self.W_Q(x)       # [B, N, d_model]
        K = self.W_K(x)
        V = self.W_V(x)
        Ew = self.W_Ew(e_emb) # [E, d_model]
        Ev = self.W_Ev(e_emb) # [E, d_model]

        src, dst = edge_index # [E], [E]
        # (B차원 지원 위해 반복)


        # 2. Gather Q, K, V for each edge (batch aware)
        # batch_index: [E], src/dst: [E]
        q_src = Q[batch_index, src, :]  # [E, d_model]
        k_dst = K[batch_index, dst, :]  # [E, d_model]
        v_dst = V[batch_index, dst, :]  # [E, d_model]

        # 3. Attention logit (GRIT: Q + K + Ew)
        att_raw = q_src + k_dst + Ew  # [E, d_model]
        att_logit = self.W_A(att_raw).squeeze(-1) / (self.d_h ** 0.5)  # [E]

        # 4. Softmax normalization per (batch, dst)
        att_exp = torch.exp(att_logit)  # [E]
        # dst_unique = (batch_index, dst) 쌍으로 그룹화
        group = batch_index * N + dst   # [E], 각 배치별 dst 노드마다 unique index
        att_sum = scatter_add(att_exp, group, dim=0, dim_size=B*N)  # [B*N]
        att_sum_dst = att_sum[group]    # [E]
        alpha = att_exp / (att_sum_dst + 1e-9)  # [E], 소프트맥스 가중치

        # 5. 메시지 (value + edge-embedding)
        msg = alpha.unsqueeze(-1) * (v_dst + Ev)  # [E, d_model]

        # 6. Aggregate message to each (batch, src)
        group_src = batch_index * N + src
        out_flat = scatter_add(msg, group_src, dim=0, dim_size=B*N)  # [B*N, d_model]
        out = out_flat.view(B, N, d)  # [B, N, d_model]
        return out



class STGRIT(nn.Module):
    def __init__(
        self,
        num_nodes,
        kernel_size,
        edge_index,
        d_edge,
        in_steps=12,
        out_steps=3,
        steps_per_day=480,
        input_dim=3,
        output_dim=3,

        tod_embedding_dim=24,
        dow_embedding_dim=24,
        adaptive_embedding_dim=80,

        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        K=3,
        
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        
        #converted_model_dim
        self.model_dim = (
            + tod_embedding_dim
            + dow_embedding_dim
            + adaptive_embedding_dim
        )


        self.use_mixed_proj = use_mixed_proj #??

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
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



        self.lin_layers_t = nn.ModuleList([
        DLinearTemporal(
        in_steps=self.in_steps,
        out_steps=self.in_steps,   # keep same length so stacking works
        num_nodes=self.num_nodes,
        individual=False,
        kernel_size=kernel_size,
        )
        for _ in range(num_layers)
        ])

        #Spatial Attention module
        self.edge_index = edge_index       # [2, E]
        self.rrwp = rrwp_encoding(self.edge_index, self.num_nodes, K=K)     # [E, K+1]
        self.e_emb = RRWPEmbedding(d_edge, K = 3)(self.rrwp)                    # [E, d_edge]
        self.num_heads = num_heads
        self.num_layers = num_layers

        #Spatial GritAttention
        self.grit_attn_layers = nn.ModuleList([
            GRITAttention(self.model_dim, d_edge, num_heads)
            for _ in range(num_layers)
        ])

        
        def forward(self, x, edge_index=None):     
        
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

            if self.adaptive_embedding_dim > 0:
                adp_emb = self.adaptive_embedding.expand(
                    size=(batch_size, *self.adaptive_embedding.shape)
                )
                features.append(adp_emb)

            
            x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

            for lin in self.lin_layers_t: 
                x = lin(x, dim=1)