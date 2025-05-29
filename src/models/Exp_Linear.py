#STAEFORMER

import torch.nn as nn
import torch
from torchinfo import summary
import numpy as np
from torch_geometric.data import Data, Batch

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
    def __init__(self, in_steps, out_steps, num_nodes,kernel_size, individual=True):
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


import math
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
print(torch_scatter.__version__)
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer



class ExphormerAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, use_bias, dim_edge=None, use_virt_nodes=False):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not dividable by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias

        if dim_edge is None:
            dim_edge = in_dim

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(dim_edge, self.out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     xavier_uniform_(self.Q)
    #     xavier_uniform_(self.K)
    #     xavier_uniform_(self.V)
    #     xavier_uniform_(self.E)

    def propagate_attention(self, batch, edge_index):
        src = batch.K_h[edge_index[0].to(torch.long)]  # (num edges) x num_heads x out_dim
        dest = batch.Q_h[edge_index[1].to(torch.long)]  # (num edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[edge_index[0].to(torch.long)] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_index[1], dim=0, out=batch.wV, reduce='add')

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.V_h.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, edge_index[1], dim=0, out=batch.Z, reduce='add')

    def forward(self, batch):
        edge_attr = batch.expander_edge_attr
        edge_index = batch.expander_edge_index
        h = batch.x
        num_node = batch.batch.shape[0]
        if self.use_virt_nodes:
            h = torch.cat([h, batch.virt_h], dim=0)
            edge_index = torch.cat([edge_index, batch.virt_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr, batch.virt_edge_attr], dim=0)
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        E = self.E(edge_attr)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = E.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch, edge_index)

        h_out = batch.wV / (batch.Z + 1e-6)

        h_out = h_out.view(-1, self.out_dim * self.num_heads)

        batch.virt_h = h_out[num_node:]
        h_out = h_out[:num_node]

        return h_out


register_layer('Exphormer', ExphormerAttention)

def get_activation(activation):
    if activation == 'relu':
        return 2, nn.ReLU()
    elif activation == 'gelu':
        return 2, nn.GELU()
    elif activation == 'silu':
        return 2, nn.SiLU()
    elif activation == 'glu':
        return 1, nn.GLU()
    else:
        raise ValueError(f'activation function {activation} is not valid!')


class ExphormerFullLayer(nn.Module):
    """Exphormer attention + FFN
    """

    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 dim_edge=None,
                 layer_norm=False, batch_norm=True,
                 activation = 'relu',
                 residual=True, use_bias=False, use_virt_nodes=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = ExphormerAttention(in_dim, out_dim, num_heads,
                                          use_bias=use_bias, 
                                          dim_edge=dim_edge,
                                          use_virt_nodes=use_virt_nodes)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        factor, self.activation_fn = get_activation(activation=activation)
        self.FFN_h_layer2 = nn.Linear(out_dim * factor, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     xavier_uniform_(self.attention.Q.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.attention.K.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.attention.V.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.attention.E.weight, gain=1 / math.sqrt(2))
    #     xavier_uniform_(self.O_h.weight, gain=1 / math.sqrt(2))
    #     constant_(self.O_h.bias, 0.0)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(batch)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        # h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = self.activation_fn(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual)




class Exp_Linear(nn.Module):
    def __init__(
        self,
        num_nodes,
        kernel_size,
        # edge_attr, # This seemed to be a placeholder for dim_edge, removing
        in_steps=12,
        out_steps=3,
        steps_per_day=480,
        input_dim=3,
        output_dim=3,
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
        spatial_attn_edge_dim=1, # Added: Dimension of edge features for ExphormerAttention
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
        self.spatial_attn_edge_dim = spatial_attn_edge_dim # Store it

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

        self.attn_layers_s = nn.ModuleList(
            [
                ExphormerFullLayer(
                    in_dim = self.model_dim,
                    out_dim = self.model_dim,
                    num_heads = self.num_heads,
                    dim_edge = self.spatial_attn_edge_dim, # Corrected: pass dim_edge
                    activation='relu') # Explicitly set activation, default in ExphormerFullLayer was 'reul'
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, edge_index, edge_attr):
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


        #Temporal processing
        for lin in self.lin_layers_t: 
            x = lin(x, dim=1)

        #Spatial processing using ExphormerFulllayer    
        
        if edge_index is None:
            raise ValueError("edge_index must be provided for ExphormerAttention.")

        # Prepare edge_attr
        if edge_attr is None:
            # Create default edge attributes if none provided, matching spatial_attn_edge_dim
            _edge_attr = torch.ones((edge_index.shape[1], self.spatial_attn_edge_dim), device=x.device)
        else:
            if edge_attr.ndim == 1:
                _edge_attr = edge_attr.unsqueeze(-1)
            else:
                _edge_attr = edge_attr
            
            if _edge_attr.shape[1] != self.spatial_attn_edge_dim:
                raise ValueError(
                    f"Provided edge_attr feature dimension ({_edge_attr.shape[1]}) "
                    f"does not match model's spatial_attn_edge_dim ({self.spatial_attn_edge_dim})"
                )    
        processed_x_timesteps = []
        # x의 shape: (batch_size, in_steps, num_nodes, model_dim)
        # edge_index의 shape: (2, num_edges) - 모든 그래프가 동일한 구조를 가진다고 가정
        # _edge_attr의 shape: (num_edges, spatial_attn_edge_dim) - 모든 그래프가 동일한 엣지 특징을 가진다고 가정

        for t_step in range(x.size(1)):  # Iterate over in_steps (T) - 시간 단계(T)를 순회합니다.
            x_t = x[:, t_step, :, :]    # (B, N, model_dim) - 현재 시간 단계 t_step의 노드 특징들을 가져옵니다.
                                        # B: batch_size, N: num_nodes

            data_list = []
            for i in range(x_t.size(0)):  # Iterate over batch_size (B) - 배치 내의 각 샘플(그래프)을 순회합니다.
                # 각 샘플(그래프)에 대한 Data 객체를 생성합니다.
                data_list.append(Data(x=x_t[i],  # 현재 샘플 i, 현재 시간 t_step의 노드 특징 (N, model_dim)
                                    edge_index=edge_index, # 그래프의 연결 정보
                                    expander_edge_index=edge_index, # ExphormerAttention이 사용할 엣지 인덱스
                                    expander_edge_attr=_edge_attr)) # ExphormerAttention이 사용할 엣지 특징

            # data_list에는 현재 시간 t_step에 해당하는 batch_size개의 Data 객체들이 들어있습니다.
            # Batch.from_data_list()는 이 Data 객체 리스트를 하나의 큰 Batch 객체로 만듭니다.
            # 이 Batch 객체는 내부적으로 여러 그래프를 효율적으로 처리할 수 있도록 구성됩니다.
            # .to(x.device)는 생성된 Batch 객체를 입력 x와 동일한 디바이스(CPU 또는 GPU)로 옮깁니다.
            pyg_batch = Batch.from_data_list(data_list).to(x.device)

            # 이제 pyg_batch는 ExphormerFullLayer와 같은 PyG 기반의 GNN 레이어가 처리할 수 있는 형태가 됩니다.
            for attn_s_layer in self.attn_layers_s: # Apply each spatial attention layer
                # ExphormerFullLayer는 pyg_batch를 입력으로 받아 그래프 어텐션을 수행하고,
                # 업데이트된 노드 특징을 포함하는 pyg_batch 객체를 반환합니다.
                pyg_batch = attn_s_layer(pyg_batch)

            # GNN 레이어 처리가 끝난 후, pyg_batch.x에는 업데이트된 노드 특징들이 합쳐진 형태로 들어있습니다.
            # (B * N, model_dim) 형태일 것입니다.
            # 이를 다시 원래의 (B, N, model_dim) 형태로 복원합니다.
            updated_x_t = pyg_batch.x.view(x_t.size(0), self.num_nodes, self.model_dim) # (B, N, model_dim)
            processed_x_timesteps.append(updated_x_t)

        # 모든 시간 단계에 대한 처리가 끝나면, processed_x_timesteps 리스트에 각 시간 단계별 결과가 쌓여있습니다.
        # torch.stack을 사용하여 이들을 다시 (B, T, N, model_dim) 형태로 합칩니다.
        x = torch.stack(processed_x_timesteps, dim=1) # (B, T, N, model_dim)


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
