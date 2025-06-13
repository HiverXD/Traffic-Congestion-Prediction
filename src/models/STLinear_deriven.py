# STLinear_deriven.py (수정판)
# ---------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import xavier_uniform_

# (1) dataset_config.py 에서 edge_spd 를 불러옵니다
from dataset.dataset_config import edge_spd  # :contentReference[oaicite:1]{index=1}

# -------------------------------------------------------------------
#  1) 시계열 처리용 블록: 기존 DLinearTemporal 그대로 사용
# -------------------------------------------------------------------
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
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
    """
    DLinear 기반 시계열 처리 블록.
    각 노드별로 Trend + Season 성분을 분리하여 Linear 학습.
    """
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
        # 먼저 [B, in_steps, num_nodes, model_dim] → [B*N*D, in_steps] 형태로 변환
        x_flat = x.permute(0, 2, 3, 1).reshape(B * N * D, T)  # [B·N·D, T]
        res, mean = self.decomp(x_flat.unsqueeze(-1))         # decomp expects [*, T, 1]
        res = res.squeeze(-1); mean = mean.squeeze(-1)

        # Linear 적용
        if self.individual:
            # 노드별로 개별 Linear
            out_res = torch.stack([
                self.lin_season[n](res[B * n * D : B * (n + 1) * D, :])
                for n in range(N)
            ], dim=1)   # [B, N, D, out_steps]
            out_mean = torch.stack([
                self.lin_trend[n](mean[B * n * D : B * (n + 1) * D, :])
                for n in range(N)
            ], dim=1)  # [B, N, D, out_steps]
        else:
            out_res  = self.lin_season(res)  # [B·N·D, out_steps]
            out_mean = self.lin_trend(mean)

        out = out_res + out_mean
        out = out.view(B, N, D, self.out_steps).permute(0, 3, 1, 2)
        # → [B, out_steps, num_nodes, model_dim]
        return out


# -------------------------------------------------------------------
#  2) Hop‐Biased Multi‐Head Attention 구현
# -------------------------------------------------------------------
class HopBiasedMultiHeadAttention(nn.Module):
    """
    Hop‐Biased Multi‐Head Attention
      - num_heads = max_hop + 1 로 설정
      - Head 0: 순수 Scaled‐Dot‐Product Attention (bias 없음)
      - Head i (i >= 1): i‐hop 관계에 γ_i * mask_i 를 어텐션 스코어에 더함
    입력: x (B, T, E, d_model) 형태
    반환: (B, T, E, d_model) 형태
    """
    def __init__(self,
                 d_model,        # 모델 차원
                 num_heads,      # 총 헤드 수 = max_hop + 1
                 edge_spd,       # numpy.ndarray of shape (E, E), 최단 경로 거리
                ):
        super().__init__()
        assert d_model % num_heads == 0, "`d_model`은 `num_heads`로 나누어떨어져야 합니다."
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads

        # 1) Q/K/V, Output Projection
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 2) Head 별로 학습 가능한 Hop‐Bias 세기 γ_h
        #    - 길이가 num_heads인 파라미터 텐서
        self.hop_gamma = nn.Parameter(torch.ones(num_heads))  
        #   → 초기값 0으로 두면, 학습 초반에는 bias 영향이 없음

        # 3) edge_spd (numpy.ndarray) → torch.LongTensor
        edge_spd_t = torch.from_numpy(edge_spd).long()  # (E, E)

        # 4) max_hop 계산: num_heads = max_hop + 1
        self.max_hop = num_heads - 1

        # 5) i‐홉 마스크 생성 및 register_buffer
        #    - hop_mask_0: 전부 0 (Head 0은 bias 없음)
        zero_mask = torch.zeros_like(edge_spd_t, dtype=torch.float32)
        self.register_buffer('hop_mask_0', zero_mask)

        #    - hop_mask_i (i = 1, …, max_hop): (edge_spd == i) 인 부분만 1
        for i in range(1, self.max_hop + 1):
            mask_i = (edge_spd_t == i).float()  # (E, E)
            self.register_buffer(f"hop_mask_{i}", mask_i)

    def forward(self, x):
        """
        x: (B, T, E, d_model)
        return: (B, T, E, d_model)
        """
        B, T, E, _ = x.shape

        # 1) [B, T, E, d_model] → [B*T, E, d_model] 로 펼치기
        x_flat = x.reshape(B * T, E, self.d_model)  # (B*T, E, d_model)

        # 2) Q, K, V 계산
        Q = self.q_proj(x_flat)  # (B*T, E, d_model)
        K = self.k_proj(x_flat)  # (B*T, E, d_model)
        V = self.v_proj(x_flat)  # (B*T, E, d_model)

        # 3) 멀티헤드 분리: (B*T, E, num_heads, d_head) → (B*T, num_heads, E, d_head)
        def split_heads(z):
            return z.view(B * T, E, self.num_heads, self.d_head) \
                    .permute(0, 2, 1, 3).contiguous()

        Qh = split_heads(Q)  # (B*T, H, E, d_head)
        Kh = split_heads(K)  # (B*T, H, E, d_head)
        Vh = split_heads(V)  # (B*T, H, E, d_head)

        # 4) 스케일된 닷 프로덕트 어텐션 스코어
        #    scores: (B*T, H, E, E)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1))  # (B*T, H, E, E)
        scores = scores / math.sqrt(self.d_head)

        # 5) Hop‐Bias 추가: 각 헤드 h 별로 hop_mask_h * γ_h 를 더해줌
        for h in range(self.num_heads):
            # head h 에 대응하는 “h-hop mask”: hop_mask_h
            # (단, hop_mask_0 은 모두 0 이므로, Head0은 순수 Attention)
            mask_h = getattr(self, f"hop_mask_{h}")  # (E, E) float32
            gamma_h = self.hop_gamma[h]             # 스칼라 텐서

            # (1,1,E,E) 형태로 확장 → 바이어스 텐서 생성
            bias_h = mask_h.unsqueeze(0).unsqueeze(0) * gamma_h  # (1,1,E,E)

            # scores[:, h, :, :] 에 더하기
            scores[:, h : h + 1, :, :] = scores[:, h : h + 1, :, :] + bias_h

        # 6) Softmax → Attention Weight
        attn_weights = F.softmax(scores, dim=-1)  # (B*T, H, E, E)

        # 7) Context 계산: (B*T, H, E, d_head)
        context = torch.matmul(attn_weights, Vh)

        # 8) 헤드 결합 및 출력 프로젝션
        #    (B*T, H, E, d_head) → (B*T, E, d_model)
        context = context.permute(0, 2, 1, 3).contiguous()  # (B*T, E, H, d_head)
        context = context.view(B * T, E, self.d_model)      # (B*T, E, d_model)
        out = self.out_proj(context)                        # (B*T, E, d_model)

        # 9) (B*T, E, d_model) → (B, T, E, d_model) 복원
        out = out.view(B, T, E, self.d_model)
        return out, attn_weights


# -------------------------------------------------------------------
#  3) Hop‐Biased Self‐Attention Layer
#     (LayerNorm, Dropout, Feed‐Forward 포함)
# -------------------------------------------------------------------
class HopBiasedSelfAttentionLayer(nn.Module):
    """
    기존 SelfAttentionLayer 와 동일한 구조인데,
    AttentionLayer 대신 HopBiasedMultiHeadAttention 을 사용합니다.

    (입력: x, dim=-2 (node 차원))
    1) x → (batch_size, ..., length, model_dim) 형태로 transpose
    2) HopBiasedMultiHeadAttention(x)
    3) Dropout + LayerNorm + Feed-Forward + Dropout + LayerNorm
    4) 다시 원래 차원으로 transpose
    """
    def __init__(self,
                 model_dim,
                 feed_forward_dim,
                 num_heads,
                 dropout,
                 edge_spd_numpy  # numpy.ndarray (E,E)
                ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.edge_spd = edge_spd_numpy

        # 1) Hop‐Biased Multi‐Head Attention
        self.attn = HopBiasedMultiHeadAttention(
            d_model   = model_dim,
            num_heads = num_heads,
            edge_spd  = edge_spd_numpy
        )

        # 2) Feed‐Forward Network 부분
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim)
        )

        # 3) LayerNorm & Dropout
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2, return_attn : bool =False) -> torch.Tensor:
        """
        x: (batch_size, ..., length, model_dim) 형태로 가정
           여기서는 주로 length=E (엣지 수)를 기준으로 Self‐Attention 수행
        """
        # 1) dim 차원을 -2로 보내기 (예: x.shape = [B, T, E, model_dim], dim=2)
        x_trans = x.transpose(dim, -2)
        # x_trans: (batch_size, ..., length, model_dim)
        residual = x_trans

        # 2) Hop‐Biased Attention
        out_attn, attn_map = self.attn(x_trans)  # (batch_size, ..., length, model_dim)
        out_attn = self.dropout1(out_attn)
        out_attn = self.ln1(residual + out_attn)

        # 3) Feed‐Forward
        residual2 = out_attn
        out_ff = self.feed_forward(out_attn)
        out_ff = self.dropout2(out_ff)
        out_ff = self.ln2(residual2 + out_ff)

        # 4) 원래 차원으로 transpose 복원
        out = out_ff.transpose(dim, -2)
        
        if return_attn:

            return out, attn_map
        else:
            return out


# -------------------------------------------------------------------
#  4) 수정된 STLinear (→ STLinear_HopBiased) 전체 모델
# -------------------------------------------------------------------
class STLinear_HopBiased(nn.Module):
    """
    STLinear 구조에 'Hop‐Biased Multi‐Head Attention'을 적용한 버전.

    • 인자:
      - num_nodes: 엣지 수(E)
      - kernel_size: 시계열 decomposition 커널 크기
      - in_steps, out_steps: 인풋/아웃풋 시계열 길이
      - steps_per_day: 하루를 몇 스텝으로 나눌지
      - input_dim, output_dim: 엣지당 입력/출력 feature 차원 (대개 3)
      - input_embedding_dim: traffic feature → 이 차원으로 임베딩
      - tod_embedding_dim, dow_embedding_dim: time-of-day / day-of-week 임베딩 크기
      - spatial_embedding_dim: (사용하지 않아도 됨, 0으로 두면 스킵)
      - adaptive_embedding_dim: (사용하지 않아도 됨, 0으로 두면 스킵)
      - feed_forward_dim: spatial 블록 Feed‐Forward hidden dim
      - num_heads: 멀티헤드 개수(=max_hop+1)
      - num_layers: 시공간 블록 반복 횟수
      - dropout: Dropout 비율
      - use_mixed_proj: 마지막 예측용 Projection 방식 선택
    """
    def __init__(self,
                 num_nodes,
                 kernel_size,
                 in_steps=12,
                 out_steps=3,
                 steps_per_day=480,
                 input_dim=3,
                 output_dim=3,
                 input_embedding_dim=32,
                 tod_embedding_dim=32,
                 dow_embedding_dim=32,
                 spatial_embedding_dim=0,
                 adaptive_embedding_dim=80,
                 feed_forward_dim=256,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.1,
                 use_mixed_proj=True,
                ):
        super().__init__()

        self.num_nodes = num_nodes          # E (엣지 수)
        self.in_steps = in_steps            # 과거 스텝 수
        self.out_steps = out_steps          # 미래 스텝 수
        self.steps_per_day = steps_per_day   # 하루 스텝 개수
        self.input_dim = input_dim          # 인풋 채널 (예: 3: volume, density, flow)
        self.output_dim = output_dim        # 아웃풋 채널
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim

        # 1) 모델 차원 계산
        #    input_emb + tod_emb + dow_emb + spatial_emb + adaptive_emb
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )

        self.num_heads = num_heads          # 헤드 수(=max_hop+1)
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        # 2) 입력 투영 (input_dim → input_embedding_dim)
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        # 3) time-of-day / day-of-week 임베딩
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)

        # 4) spatial_embedding_dim > 0 이면 노드별 learnable embedding
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, spatial_embedding_dim)
            )
            xavier_uniform_(self.node_emb)

        # 5) adaptive embedding (in_steps × num_nodes × adaptive_embedding_dim)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
            )
            xavier_uniform_(self.adaptive_embedding)

        # 6) 마지막 예측용 projection
        if use_mixed_proj:
            # (num_nodes, in_steps * model_dim) → (out_steps * output_dim)
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            # (model_dim, in_steps) → (model_dim, out_steps), → (out_steps, num_nodes, output_dim)
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj   = nn.Linear(self.model_dim, self.output_dim)

        # 7) 시계열 블록: DLinearTemporal (in_steps → in_steps) × num_layers
        self.lin_layers_t = nn.ModuleList([
            DLinearTemporal(
                in_steps=self.in_steps,
                out_steps=self.in_steps,   # stacking 용도이므로 동일 길이 유지
                num_nodes=self.num_nodes,
                kernel_size=kernel_size,
                individual=False,
            )
            for _ in range(num_layers)
        ])

        # 8) 공간 블록: HopBiasedSelfAttentionLayer × num_layers
        self.attn_layers_s = nn.ModuleList([
            HopBiasedSelfAttentionLayer(
                model_dim      = self.model_dim,
                feed_forward_dim = feed_forward_dim,
                num_heads      = num_heads,
                dropout        = dropout,
                edge_spd_numpy = edge_spd  # :contentReference[oaicite:2]{index=2}
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index=None, edge_attr=None, return_attn=False):
        """
        • 입력:
          - x: (B, in_steps, E, input_dim + tod + dow = 3)
          - edge_index, edge_attr는 Trainer에서 전달되지만, HopBiasedAttention은 edge_spd만 사용
        • 출력: (B, out_steps, E, output_dim)
        """
        batch_size = x.shape[0]

        # 1) time-of-day, day-of-week 분리
        if self.tod_embedding_dim > 0:
            tod = x[..., 3] / self.steps_per_day
        if self.dow_embedding_dim > 0:
            dow = x[..., 4]

        # 2) input_dim 채널 (volume, density, flow)만 추출
        x_in = x[..., : self.input_dim]  # (B, in_steps, E, input_dim)

        # 3) input_proj → (B, in_steps, E, input_embedding_dim)
        x_feat = self.input_proj(x_in)
        features = [x_feat]

        # 4) tod embedding (B, in_steps, E, tod_embedding_dim)
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )
            features.append(tod_emb)

        # 5) dow embedding (B, in_steps, E, dow_embedding_dim)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow.long())
            features.append(dow_emb)

        # 6) spatial embedding (B, in_steps, E, spatial_embedding_dim)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)

        # 7) adaptive embedding (B, in_steps, E, adaptive_embedding_dim)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        # 8) 모든 feature concat → x (B, in_steps, E, model_dim)
        x_cat = torch.cat(features, dim=-1)

        # 9) 시계열 블록 반복
        x_seq = x_cat
        for lin in self.lin_layers_t:
            x_seq = lin(x_seq, dim=1)  # (B, in_steps, E, model_dim)

        # 10) 공간 블록 반복 (Hop‐Biased Self‐Attention)
        x_spatial = x_seq
        spatial_maps = []
        for attn in self.attn_layers_s:
            if return_attn:
                x_spatial, attn_map = attn(x_spatial, dim=2, return_attn = True)  # (B, in_steps, E, model_dim)
                spatial_maps.append(attn_map)
            else:
                x_spatial = attn(x_spatial, dim=2)
        # 11) 최종 예측
        if self.use_mixed_proj:
            # (B, in_steps, E, model_dim) → (B, E, in_steps, model_dim)
            out = x_spatial.transpose(1, 2)   # (B, E, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out)       # (B, E, out_steps * output_dim)
            out = out.view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)         # (B, out_steps, E, output_dim)
        else:
            # (B, in_steps, E, model_dim) → (B, model_dim, E, in_steps)
            out = x_spatial.transpose(1, 3)   # (B, model_dim, E, in_steps)
            out = self.temporal_proj(out)     # (B, model_dim, E, out_steps)
            out = self.output_proj(out.transpose(1, 3))  # (B, out_steps, E, output_dim)

        if return_attn:
            return out, spatial_maps
        else:
            return out


# ==========================
# STLinear SPE version
# ==========================
import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    """
    Vanilla Multi-Head Self-Attention 레이어 (Spatial Attention 용).
    • 입력 x: (B, T, E, dim) 형태로 들어오며,
      여기서 E는 엣지(또는 노드) 개수, dim은 모델 차원(d_model)입니다.
    • dim 인자에 따라 x[..., dim축]을 Spatial Attention 차원으로 간주해 처리합니다.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert dim % num_heads == 0, "`dim`은 `num_heads`로 나누어떨어져야 합니다."
        self.d_model = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        self.scale = math.sqrt(self.d_head)

        # Q, K, V를 위한 선형 계층
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        # 최종 출력 투영
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Dropout + LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, dim: int = 2, return_attn: bool = False) -> torch.Tensor:
        """
        • x: (B, T, E, dim) 형태 텐서
        • dim: Attention을 수행할 차원 ( 여기서는 기본값 2 → E 축 )
        → 출력: (B, T, E, dim)
        """
        B, T, E, D = x.shape
        # 1) dim 차원을 -2로 옮겨서 (B, T, E, dim) → (B, T, dim, E) 형태로 만듭니다.
        #    하지만 여기서는 reshaping 후 바로 (B*T, E, D)로 flatten 하므로, transpose 없이 view로 처리 가능합니다.
        x_flat = x.reshape(B * T, E, D)  # (B*T, E, dim)

        # 2) Q, K, V 선형 투영 및 헤드 분할
        #    먼저 (B*T, E, dim) → (B*T, E, num_heads, d_head), → (B*T, num_heads, E, d_head)
        Q = self.W_q(x_flat).view(B * T, E, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.W_k(x_flat).view(B * T, E, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.W_v(x_flat).view(B * T, E, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # 3) 어텐션 스코어 계산: (B*T, num_heads, E, d_head) × (B*T, num_heads, d_head, E) → (B*T, num_heads, E, E)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B*T, H, E, E)

        # 4) 소프트맥스
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B*T, H, E, E)

        # 5) 어텐션 결과: (B*T, H, E, E) × (B*T, H, E, d_head) → (B*T, H, E, d_head)
        attn_out = torch.matmul(attn_weights, V)  # (B*T, H, E, d_head)

        # 6) 헤드 결합: (B*T, H, E, d_head) → (B*T, E, dim)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B * T, E, D)

        # 7) 최종 투영 및 Residual + LayerNorm
        out = self.out_proj(attn_out)                     # (B*T, E, dim)
        out = self.norm(x_flat + self.dropout(out))       # (B*T, E, dim)

        # 8) 원래 차원 (B, T, E, dim)으로 복원하여 반환

        if return_attn:
            return out.view(B, T, E, D), attn_weights.view(B, T, self.num_heads, E, E)
        else:
            return out.view(B, T, E, D)

class SPEBiasedMultiHeadAttention(nn.Module):
    """
    d_model 차원이 num_heads로 나누어떨어져야 하며,
    spe: (E, p) SPE 텐서를 받아 (E×E) 형태의 bias를 계산해 로짓에 더함.
    """
    def __init__(self, d_model: int, num_heads: int, spe: torch.Tensor):
        super().__init__()
        assert d_model % num_heads == 0, "`d_model`은 `num_heads`로 나누어떨어져야 합니다."
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.p = spe.size(1)  # SPE 차원

        # 1) Q, K, V 프로젝션
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # 2) SPE 기반 bias: (E, p) spe → (E, E) bias
        #    각 헤드 동일하게 사용 가능
        self.register_buffer('attn_bias', (spe @ spe.t()) / (spe.size(1) ** 0.5))

        # 3) scale 상수
        self.scale = (self.d_head) ** 0.5

    def forward(self, x: torch.Tensor):
        """
        x: (B*T, E, d_model) 형태
        returns: (B*T, E, d_model), (B*T, num_heads, E, E)
        """
        B_T, E, D = x.shape
        # Q, K, V 투영
        Q = self.W_q(x).view(B_T, E, self.num_heads, self.d_head).permute(0, 2, 1, 3)  # (B*T, H, E, d_head)
        K = self.W_k(x).view(B_T, E, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.W_v(x).view(B_T, E, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # Q·Kᵀ 계산
        attn_logits = (Q @ K.transpose(-2, -1)) / self.scale  # (B*T, H, E, E)

        # SPE 기반 bias 추가
        bias = self.attn_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, E, E)
        attn_logits = attn_logits + bias

        # Softmax
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B*T, H, E, E)

        # V에 곱해서 context 구함
        attn_out = attn_weights @ V       # (B*T, H, E, d_head)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B_T, E, D)  # (B*T, E, d_model)

        # 최종 projection
        attn_out = self.out_proj(attn_out)  # (B*T, E, d_model)
        return attn_out, attn_weights

class SPEBiasedSelfAttentionLayer(nn.Module):
    """
    SPE 기반 bias를 로직에 추가한 Spatial Self-Attention 레이어.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float, spe: torch.Tensor):
        super().__init__()
        self.mha = SPEBiasedMultiHeadAttention(d_model=dim, num_heads=num_heads, spe=spe)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, dim: int = 2):
        B, T, E, D = x.shape
        x_flat = x.view(B * T, E, D)                    # (B*T, E, D)
        attn_out, _ = self.mha(x_flat)                  # (B*T, E, D)
        out = self.norm(x_flat + self.dropout(attn_out))  # Residual + Norm
        return out.view(B, T, E, D)                     # (B, T, E, D)


# ------------------------------------------------------------------------------
# 1) Spectral Positional Encoding (SPE) 계산 함수
# ------------------------------------------------------------------------------
def compute_spe_torch(edge_adj_mat: torch.Tensor, p: int, normalized: bool = False) -> torch.Tensor:
    """
    PyTorch를 사용한 (dense) Spectral Positional Encoding 계산 함수.

    • edge_adj_mat: (E, E) 형태의 0/1 인접행렬을 담은 torch.Tensor (float32 또는 float64)
    • p: 추출할 고유벡터 개수 (상수 모드 제외한 차원)
    • normalized: True면 정규화 라플라시안 사용, False면 비정규화 라플라시안
    ------------------------------------------
    returns: SPE 텐서, shape (E, p), dtype=torch.float32
    """
    # 1) 인접행렬이 NumPy 배열 형태일 경우, torch.Tensor로 변환 필요
    #    이미 torch.Tensor라면 이 과정을 생략해도 됩니다.
    if not isinstance(edge_adj_mat, torch.Tensor):
        A = torch.from_numpy(edge_adj_mat).float()
    else:
        A = edge_adj_mat.float()

    # 2) GPU 사용 가능한 경우, A를 GPU로 옮김
    device = A.device
    if torch.cuda.is_available():
        A = A.to('cuda')

    E = A.size(0)
    # 3) 차수행렬 D 계산
    deg = torch.sum(A, dim=1)  # (E,)

    if normalized:
        # 정규화 라플라시안 L_sym = I - D^{-1/2} A D^{-1/2}
        deg_root_inv = torch.pow(deg + 1e-8, -0.5)  # (E,)
        D_root_inv = torch.diag(deg_root_inv)  # (E, E)
        L = torch.eye(E, device=A.device) - D_root_inv @ A @ D_root_inv
    else:
        # 비정규화 라플라시안 L = D - A
        D = torch.diag(deg)  # (E, E)
        L = D - A           # (E, E)

    # 4) 대칭행렬 L에 대해 고유분해 수행
    #    torch.linalg.eigh는 eigenvalues/eigenvectors를 모두 반환 (밀집 행렬)
    eigvals, eigvecs = torch.linalg.eigh(L)  # eigvals: (E,), eigvecs: (E, E)

    # 5) 가장 작은 고유값(상수 모드)을 제외하고, 다음 p개 벡터를 SPE로 사용
    #    (eigvecs가 오름차순 정렬되어 있어서 eigvecs[:, 0]은 상수 모드)
    spe = eigvecs[:, 1 : p + 1].clone()  # (E, p)

    # 6) 열별 정규화: 평균 0, 분산 1
    spe = (spe - spe.mean(dim=0, keepdim=True)) / (spe.std(dim=0, keepdim=True) + 1e-6)

    # 7) CPU/GPU 반환: 원래 edge_adj_mat이 있던 device로 맞춰 보내기
    spe = spe.to(device)

    return spe  # (E, p)

def compute_spe_torch(edge_adj_mat: torch.Tensor, p: int, normalized: bool = False) -> torch.Tensor:
    """
    PyTorch 기반 SPE 계산 함수 (dense eigen-decomposition).
    """
    if not isinstance(edge_adj_mat, torch.Tensor):
        A = torch.from_numpy(edge_adj_mat).float()
    else:
        A = edge_adj_mat.float()

    # 인접행렬이 CPU/Tensor라면 우선 device 기억
    original_device = A.device

    # eigen-decomposition은 GPU에서 수행하면 더 빠릅니다.
    if torch.cuda.is_available():
        A = A.to('cuda')

    E = A.size(0)
    deg = torch.sum(A, dim=1)  # (E,)

    if normalized:
        deg_root_inv = torch.pow(deg + 1e-8, -0.5)  # (E,)
        D_root_inv = torch.diag(deg_root_inv)       # (E, E)
        L = torch.eye(E, device=A.device) - D_root_inv @ A @ D_root_inv
    else:
        D = torch.diag(deg)  # (E, E)
        L = D - A             # (E, E)

    eigvals, eigvecs = torch.linalg.eigh(L)       # eigvals: (E,), eigvecs: (E, E)
    spe = eigvecs[:, 1 : p + 1].clone()            # (E, p)
    spe = (spe - spe.mean(dim=0, keepdim=True)) / (spe.std(dim=0, keepdim=True) + 1e-6)

    # 원래 device(CPU 또는 GPU)로 복귀
    spe = spe.to(original_device)

    return spe  # (E, p)


# ② STLinear_SPE 클래스
class STLinear_SPE(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 kernel_size: int,
                 in_steps: int = 12,
                 out_steps: int = 3,
                 steps_per_day: int = 480,
                 input_dim: int = 3,
                 output_dim: int = 3,
                 input_embedding_dim: int = 32,
                 tod_embedding_dim: int = 32,
                 dow_embedding_dim: int = 32,
                 spatial_embedding_dim: int = 0,
                 adaptive_embedding_dim: int = 0,
                 spe_dim: int = 32,
                 spe_out_dim: int = 96,   # ← SPE를 투영할 차원
                 feed_forward_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_mixed_proj: bool = True,
                 normalized_laplacian: bool = False,
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
        self.spe_dim = spe_dim
        self.spe_out_dim = spe_out_dim        # ← SPE 투영 후 차원
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        # ────────────────────────────────────────────────────────────────────
        # (1) 모델 전체 차원 정의: 원래 임베딩 합 + SPE 투영 차원
        # ────────────────────────────────────────────────────────────────────
        #   original_embedding_sum = input_emb + tod_emb + dow_emb + spatial_emb + adaptive_emb
        #   => 예시: 32 + 32 + 32 + 0 + 0 = 96
        #
        #   최종 모델 차원(model_dim)은 여기에 SPE 투영 차원(spe_out_dim)을 더한 것 (예시: 96 + 96 = 192)
        self.original_embedding_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )  # 예: 96
        self.model_dim = self.original_embedding_dim + spe_out_dim  # 예: 96 + 96 = 192

        # ────────────────────────────────────────────────────────────────────
        # (2) 입력 피처 투영 (input_dim → input_embedding_dim)
        # ────────────────────────────────────────────────────────────────────
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        # ────────────────────────────────────────────────────────────────────
        # (3) TOD, DOW 임베딩
        # ────────────────────────────────────────────────────────────────────
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)

        # ────────────────────────────────────────────────────────────────────
        # (4) Spatial 임베딩
        # ────────────────────────────────────────────────────────────────────
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)

        # ────────────────────────────────────────────────────────────────────
        # (5) Adaptive 임베딩
        # ────────────────────────────────────────────────────────────────────
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        # ────────────────────────────────────────────────────────────────────
        # (6) SPE 버퍼 + SPE → spe_out_dim 투영
        # ────────────────────────────────────────────────────────────────────
        #   - _spe_buffer: 나중에 load_spe로 덮어씌움
        #   - spe_proj: (spe_dim → spe_out_dim)
        self.register_buffer('_spe_buffer', torch.zeros(num_nodes, spe_dim))
        self.spe_proj = nn.Linear(spe_dim, spe_out_dim)  # (E, spe_dim) → (E, spe_out_dim)

        # ────────────────────────────────────────────────────────────────────
        # (7) 예측용 Projection
        # ────────────────────────────────────────────────────────────────────
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim,
                out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj   = nn.Linear(self.model_dim, self.output_dim)

        # ────────────────────────────────────────────────────────────────────
        # (8) 시계열 블록: DLinearTemporal × num_layers
        # ────────────────────────────────────────────────────────────────────
        self.lin_layers_t = nn.ModuleList([
            DLinearTemporal(
                in_steps=self.in_steps,
                out_steps=self.in_steps,
                num_nodes=self.num_nodes,
                kernel_size=kernel_size,
                individual=False
            )
            for _ in range(num_layers)
        ])

        # ────────────────────────────────────────────────────────────────────
        # (9) 공간 블록: 일반 Self-Attention Layer × num_layers
        # ────────────────────────────────────────────────────────────────────
        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(
                dim=self.model_dim,     # ← 여기 model_dim=192
                num_heads=self.num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # ────────────────────────────────────────────────────────────────────
        # (10) Dropout
        # ────────────────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ────────────────────────────────────────────────────────────────────
        # (11) 정규화 라플라시안 여부
        # ────────────────────────────────────────────────────────────────────
        self.normalized_laplacian = normalized_laplacian

    def load_spe(self, edge_adj_mat: torch.Tensor):
        """
        • edge_adj_mat: (E, E) 형태의 0/1 인접행렬 (torch.Tensor 또는 NumPy)
        • compute_spe_torch를 호출해 SPE를 계산 후 '_spe_buffer'에 저장
        """
        spe_tensor = compute_spe_torch(edge_adj_mat, self.spe_dim, self.normalized_laplacian)
        spe_tensor = spe_tensor.to(self._spe_buffer.device)
        self._spe_buffer = spe_tensor  # SPE 텐서 덮어쓰기

    def forward(self, x: torch.Tensor, edge_index=None, edge_attr=None, return_attn: bool = False):
        """
        • x: (B, in_steps, E, input_dim + 2) 형태 (input_dim=3 → 마지막 두 채널이 TOD,DOW)
        • edge_index, edge_attr: Trainer에서 넘겨주지만, SPE 모델은 실제로 사용하지 않음
        • returns: (B, out_steps, E, output_dim)
        """
        B = x.shape[0]

        # 1) TOD / DOW 분리
        if self.tod_embedding_dim > 0:
            tod = x[..., self.input_dim] / self.steps_per_day
        if self.dow_embedding_dim > 0:
            dow = x[..., self.input_dim + 1]

        # 2) traffic feature만 추출
        x_in = x[..., : self.input_dim]  # (B, in_steps, E, input_dim)

        # 3) input_proj → (B, in_steps, E, input_embedding_dim)
        x_feat = self.input_proj(x_in)
        features = [x_feat]

        # 4) TOD 임베딩 추가
        if self.tod_embedding_dim > 0:
            tod_idx = (tod * self.steps_per_day).long()  # (B, in_steps, E)
            tod_emb = self.tod_embedding(tod_idx)        # (B, in_steps, E, tod_dim)
            features.append(tod_emb)

        # 5) DOW 임베딩 추가
        if self.dow_embedding_dim > 0:
            dow_idx = dow.long()                         # (B, in_steps, E)
            dow_emb = self.dow_embedding(dow_idx)        # (B, in_steps, E, dow_dim)
            features.append(dow_emb)

        # 6) Spatial 임베딩 추가
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.unsqueeze(0).unsqueeze(1) \
                          .expand(B, self.in_steps, -1, -1)       # (B, in_steps, E, spatial_dim)
            features.append(spatial_emb)

        # 7) Adaptive 임베딩 추가
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.unsqueeze(0) \
                       .expand(B, -1, -1, -1)                    # (B, in_steps, E, adaptive_dim)
            features.append(adp_emb)

        # 8) SPE 임베딩 추가
        #    - _spe_buffer: (E, spe_dim)
        spe_emb = self.spe_proj(self._spe_buffer)             # → (E, spe_out_dim)
        spe_emb = spe_emb.unsqueeze(0).unsqueeze(1)           # → (1, 1, E, spe_out_dim)
        features.append(spe_emb.expand(B, self.in_steps, -1, -1))  # → (B, in_steps, E, spe_out_dim)

        # 9) 모든 feature concat → x_cat: (B, in_steps, E, model_dim)
        x_cat = torch.cat(features, dim=-1)  # model_dim = original_embedding_dim + spe_out_dim = 192

        # 10) 시계열 블록 반복
        x_seq = x_cat
        for lin in self.lin_layers_t:
            x_seq = lin(x_seq, dim=1)  # (B, in_steps, E, model_dim)

        # 11) 공간 블록 반복 (Self-Attention)
        x_spatial = x_seq
        attention_maps = [] if return_attn else None
        for attn in self.attn_layers_s:
            if return_attn:
                x_spatial, attn_weights = attn(x_spatial, dim=2, return_attn=True)
                attention_maps.append(attn_weights)
            else:
                x_spatial = attn(x_spatial, dim=2)

        # 12) 최종 예측
        if self.use_mixed_proj:
            out = x_spatial.transpose(1, 2)                 # → (B, E, in_steps, model_dim)
            out = out.reshape(B, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out)                     # → (B, E, out_steps * output_dim)
            out = out.view(B, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)                       # → (B, out_steps, E, output_dim)
        else:
            out = x_spatial.transpose(1, 3)                 # → (B, model_dim, E, in_steps)
            out = self.temporal_proj(out)                   # → (B, model_dim, E, out_steps)
            out = self.output_proj(out.transpose(1, 3))     # → (B, out_steps, E, output_dim)

        if return_attn:
            return out, attention_maps  # attention_maps: List of (B, T, H, E, E)
        else:
            return out
