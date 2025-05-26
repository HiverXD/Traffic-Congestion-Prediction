#STAEFORMER

import torch.nn as nn
import torch
from torchinfo import summary

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
    def __init__(self, in_steps, out_steps, num_nodes, individual=True, kernel_size):
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
                 activation = 'reul',
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


class custom_model(nn.Module):
    def __init__(
        self,
        num_nodes,
        kernel_size,
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
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x,edge_index=None,edge_attr=None):
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

        for lin in self.lin_layers_t: 
            x = lin(x, dim=1)
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


if __name__ == "__main__":
    model = custom_model(50, 12, 3)
    summary(model, [64, 12, 207, 3])