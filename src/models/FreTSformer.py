#STAEFORMER

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torchinfo import summary
import numpy as np

class FreTS(nn.Module):
    def __init__(self, in_steps, out_steps, num_nodes, input_dim):
        super().__init__()
        self.embed_size = 128
        self.hidden_size = 256
        self.pre_length = out_steps
        self.seq_length = in_steps

        # 시공간 그래프 관련 변수 추가 및 재정의
        self.num_nodes = num_nodes
        self.input_dim = input_dim # 각 노드당 기본 특성 차원 (input_dim + tod + dow)

        # self.feature_size는 기존 코드의 'Channel' 개념과 유사하게 사용됩니다.
        # 이제 각 노드의 각 input_dim 특성이 독립적인 '채널'처럼 처리됩니다.
        self.feature_size = self.input_dim

        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, In_steps, Num_nodes, Input_dim]
        B, T, N, D_in = x.shape

        # 목표: [B, (N*D_in), T, D_embed] 형태로 만들기
        # 1. [B, N, D_in, T]로 permute
        x = x.permute(0, 2, 3, 1)

        # 2. N과 D_in 차원을 합쳐서 (N*D_in) 크기의 새로운 '채널' 차원 생성
        x = x.reshape(B, N * D_in, T)

        # 3. 임베딩을 위해 마지막 차원 (T) 옆에 1을 추가 [B, N*D_in, T, 1]
        x = x.unsqueeze(3)

        # self.embeddings: [1, embed_size]
        y = self.embeddings
        
        # 브로드캐스팅을 통해 [B, N*D_in, T, embed_size] 결과 생성
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N_channels_total, L):
        # x: [B, N_channels_total, T, D_embed]
        x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L (time) dimension
        y = self.FreMLP(B, N_channels_total, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # frequency-domain MLPs (FreMLP는 입력 차원만 잘 들어오면 내부 로직 변경 불필요)
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, In_steps (T), Num_nodes (N), Input_dim (D_in)]
        B, T, N, D_in = x.shape

        # embedding x: [B, (N*D_in), T, D_embed]
        x = self.tokenEmb(x)
        bias = x

        # 시간 축 연산만 수행
        # N 인자에 (Num_nodes * Input_dim)을 전달
        x = self.MLP_temporal(x, B, N * D_in, T)

        x = x + bias

        # Final FC layer for prediction
        # x: [B, (N*D_in), T, D_embed]
        # reshape: [B, (N*D_in), T*D_embed]
        x = self.fc(x.reshape(B, N * D_in, -1))
        # Output: [B, (N*D_in), pre_length]

        # 최종 출력 형태를 [Batch, pre_length, num_nodes, input_dim]으로 재구성
        # 1. pre_length를 두 번째 차원으로 가져오기
        x = x.permute(0, 2, 1) # [B, pre_length, (N*D_in)]

        # 2. Num_nodes와 Input_dim으로 다시 분리
        x = x.reshape(B, self.pre_length, N, D_in)

        return x








class StaticSpexphormerAttention(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_heads, use_bias=False):

        super().__init__()


        
        if out_dim % num_heads != 0:

            raise ValueError('hidden dimension is not dividable by the number of heads')

        self.out_dim = out_dim // num_heads

        self.num_heads = num_heads



        # layer_idx 관련 속성 제거

        # self.edge_index_name = f'edge_index_layer_{layer_idx}'

        # self.edge_attr_name = f'edge_type_layer_{layer_idx}'



        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

        self.E1 = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

        self.E2 = nn.Linear(in_dim, num_heads, bias=True)

        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)



    def forward(self, x, edge_index, edge_attr, num_nodes_in_layer=None, edge_embeddings=None):

        # n1, n2는 num_nodes_in_layer를 통해 명시적으로 전달되거나, x의 크기에서 유추

        # 정적 그래프에서는 단일 레이어이므로, n1 == n2 == x.shape[0] 일 가능성이 높습니다.

        # 여기서는 편의상 x.shape[0]을 n1, n2로 사용합니다.

        # 만약 특정 계층 구조가 여전히 필요하다면 num_nodes_in_layer를 활용할 수 있습니다.

        n_nodes = x.shape[0] # 전체 노드 수



        # 기존 코드에서 n1 == batch.x.shape[0] 이었으므로, n1은 전체 노드 수

        # n2는 다음 레이어의 노드 수였으나, 정적 그래프에서는 n2도 n_nodes가 될 것입니다.

        # Q_h는 다음 레이어의 노드에 대한 쿼리이므로 n2를 사용했으나,

        # 정적 그래프에서는 현재 레이어의 모든 노드가 다음 레이어의 입력이 될 수 있습니다.

        # 따라서, Q_h도 모든 노드 (n_nodes)에 대해 계산하도록 변경합니다.

        Q_h = self.Q(x).view(-1, self.num_heads, self.out_dim)

        K_h = self.K(x).view(-1, self.num_heads, self.out_dim)

        V_h = self.V(x).view(-1, self.num_heads, self.out_dim)



        if cfg.dataset.edge_encoder_name == 'TypeDictEdge2' and edge_embeddings is not None:

            # edge_embeddings가 인자로 직접 전달되어야 함

            E1 = self.E1(edge_embeddings)[edge_attr].view(n_nodes, -1, self.num_heads, self.out_dim)

            E2 = self.E2(edge_embeddings)[edge_attr].view(n_nodes, -1, self.num_heads, 1)

        else:

            # edge_attr이 직접 엣지 특징 벡터인 경우

            E1 = self.E1(edge_attr).view(n_nodes, -1, self.num_heads, self.out_dim)

            E2 = self.E2(edge_attr).view(n_nodes, -1, self.num_heads, 1)

        

        # neighbors = edge_index[0, :] 는 PyG의 coo 형식에서 source 노드를 의미합니다.

        # 이웃 노드의 K, V를 가져오기 위해 edge_index[0] (source 노드 인덱스)를 사용합니다.

        neighbors = edge_index[0, :]

        

        # deg 계산은 여전히 필요합니다.

        # 여기서는 edge_index[0].shape[0]이 총 엣지 수이므로,

        # 이를 n_nodes로 나누어 각 노드당 평균 이웃 수를 추정합니다.

        # 하지만 더 정확하게는 각 타겟 노드(n_nodes)에 연결된 source 노드들의 최대 수(degree)가 필요합니다.

        # 이 코드는 규칙적인 그래프를 가정하거나, edge_index가 이미 (target_node, deg * num_heads) 형태로 reshape 되어 있다고 가정합니다.

        # 정적 그래프에서 'deg'가 의미하는 바를 정확히 맞춰야 합니다.

        # 원본 코드에서 neighbors.shape[0] // n2는 (총 엣지 수) // (타겟 노드 수) 였습니다.

        # 여기서 n2는 사실상 현재 레이어의 노드 수 (n_nodes)와 같다고 볼 수 있으므로,

        # deg는 각 노드가 가진 이웃의 수 (여기서는 수신 엣지의 수)의 평균 또는 균일한 최대값으로 간주됩니다.

        # 만약 그래프가 불규칙적이라면, 이 부분을 다른 방식으로 처리해야 할 수 있습니다.

        # 여기서는 원본 코드의 로직을 최대한 유지합니다.

        

        # edge_index의 타겟 노드를 기반으로 deg를 계산하는 것이 더 일반적입니다.

        # 하지만 원본 코드가 neighbors = edge_index[0, :]로 시작했으므로,

        # 이웃의 수는 각 엣지의 source 노드의 수로 간주됩니다.

        

        # PyG의 MessagePassing 구현처럼 인접 행렬을 통한 메시지 전달이 아닐 경우

        # deg는 각 노드에 연결된 엣지의 수 (혹은 최대 엣지 수)를 의미합니다.

        # 여기서는 neighbors = edge_index[0, :] 이므로,

        # deg는 각 n_nodes가 가지는 이웃 (source 노드)의 수로 해석됩니다.

        # reshape(n_nodes, deg)를 보면, 각 타겟 노드(n_nodes)가 deg개의 이웃을 가진다고 가정합니다.

        # 만약 그래프가 불규칙적이고 패딩이 없다면 이 부분에서 문제가 발생할 수 있습니다.

        

        # 가정: 모든 타겟 노드 n_nodes가 deg 개의 이웃을 가짐 (패딩 등으로 맞췄다고 가정)

        deg = neighbors.shape[0] // n_nodes

        neighbors = neighbors.reshape(n_nodes, deg) # [num_nodes, deg]



        # K_h와 V_h는 이웃 노드 (neighbors)로부터 가져옵니다.

        # K_h.shape: [num_nodes, num_heads, out_dim] -> [neighbors.shape[0], num_heads, out_dim]

        # neighbors는 [n_nodes, deg] 이므로, K_h[neighbors]는 [n_nodes, deg, num_heads, out_dim]

        K_h = K_h[neighbors] # [n_nodes, deg, num_heads, out_dim]

        V_h = V_h[neighbors] # [n_nodes, deg, num_heads, out_dim]

        

        # E1도 [n_nodes, deg, num_heads, out_dim] 형태여야 함

        # 현재 E1은 [n_nodes, -1, num_heads, out_dim] 이므로, -1이 deg와 일치해야 합니다.

        # E1의 두 번째 차원을 deg로 가정합니다.

        E1 = E1.view(n_nodes, deg, self.num_heads, self.out_dim) # [n_nodes, deg, num_heads, out_dim]



        # 어텐션 스코어 계산: E1 * K_h (요소별 곱셈)

        # 결과: [n_nodes, deg, num_heads, out_dim]

        score = torch.mul(E1, K_h)

        

        # Q_h는 [n_nodes, num_heads, out_dim]

        # score는 [n_nodes, deg, num_heads, out_dim]

        # Q_h와 score를 bmm 하기 위해 차원 재조정

        # score.view(-1, deg, self.out_dim) -> [n_nodes * num_heads, deg, out_dim]

        # Q_h.view(-1, self.out_dim, 1) -> [n_nodes * num_heads, out_dim, 1]

        

        # [n_nodes * num_heads, deg, out_dim] @ [n_nodes * num_heads, out_dim, 1]

        # -> [n_nodes * num_heads, deg, 1]

        score = torch.bmm(score.permute(0, 2, 1, 3).reshape(-1, deg, self.out_dim), 

                          Q_h.permute(0, 1, 2).reshape(-1, self.out_dim, 1))

        

        score = score.view(n_nodes, self.num_heads, deg) # [n_nodes, num_heads, deg]



        # E2는 [n_nodes, -1, num_heads, 1] (여기서 -1은 deg)

        # E2.squeeze(-1) -> [n_nodes, deg, num_heads]

        # .permute([0, 2, 1]) -> [n_nodes, num_heads, deg]

        E2 = E2.view(n_nodes, deg, self.num_heads, 1) # [n_nodes, deg, num_heads, 1]

        score = score + E2.squeeze(-1).permute([0, 2, 1])

        

        score = score.clamp(-8, 8)

        score = F.softmax(score, dim=-1) # [n_nodes, num_heads, deg]

        

        # V_h: [n_nodes, deg, num_heads, out_dim]

        # V_h.permute(0, 2, 1, 3) -> [n_nodes, num_heads, deg, out_dim]

        V_h = V_h.permute(0, 2, 1, 3)

        

        # score: [n_nodes, num_heads, deg] -> score.unsqueeze(-1) -> [n_nodes, num_heads, deg, 1]

        score = score.unsqueeze(-1)

        

        # 요소별 곱셈: [n_nodes, num_heads, deg, 1] * [n_nodes, num_heads, deg, out_dim]

        # 결과: [n_nodes, num_heads, deg, out_dim]

        h_out = torch.mul(score, V_h)

        

        # Sum over deg dimension: [n_nodes, num_heads, out_dim]

        h_out = h_out.sum(dim=2)

        

        # Flatten to [n_nodes, num_heads * out_dim]

        h_out = h_out.reshape(n_nodes, -1)



        return h_out

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


class FreTSformer(nn.Module):
    def __init__(
        self,
        num_nodes,
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

        

        #시간 축 연산
        self.lin_layers_t = nn.ModuleList(
            [
                FreTS(
                self.in_steps,
                self.in_steps,   # keep same length so stacking works
                self.num_nodes,
                self.model_dim,
            )
            for _ in range(num_layers)
        ])

        #공간 축 연산
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
            x = lin(x)
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