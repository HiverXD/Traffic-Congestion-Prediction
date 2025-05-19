# src/utils/edge_prediction_visualization.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

def add_tod_dow(raw_data: np.ndarray, week_steps: int, C_origin: int) -> np.ndarray:
    """
    원본 데이터에 시간(tod)·요일(dow) 채널을 추가합니다.

    Args:
        raw_data: np.ndarray, shape (T_total, E, C_orig)
            - T_total: 전체 타임스텝 수
            - E: 엣지(센서) 수
            - C_orig: 원래 피처 채널 수 (예: volume, density, flow)
        week_steps: int, default=480*7
            - 주 단위 전체 스텝 수 (예: 7일간 1시간당 1스텝으로 480스텝을 가정)
        C_origin: int, default=3
            - 원본 데이터의 채널 수. 차원이 맞지 않을 경우를 검증하기 위해 사용

    Returns:
        np.ndarray, shape (T_total, E, C_orig + 2)
        - 마지막 두 채널이 각각 tod(0~24 float), dow(0~6 int→float) 특성입니다.
    """
    if raw_data.shape[2] == C_origin:
        pass
    elif raw_data.shape[2] == (C_origin+2):
        return raw_data
    else:
        raise Exception('shape error')
    
    T_total, E, C_orig = raw_data.shape
    day_steps = week_steps // 7

    # 1) 모든 타임스텝 인덱스 생성
    timesteps = np.arange(T_total)

    # 2) 시간대 특성 (0~24)
    tod = (timesteps % day_steps) * (24.0 / day_steps)
    # 3) 요일 특성 (0~6)
    dow = (timesteps // day_steps) % 7

    # 4) (T,1,1) → (T,E,1)로 확장
    tod_feat = np.tile(tod[:, None, None], (1, E, 1)).astype(np.float32)
    dow_feat = np.tile(dow[:, None, None], (1, E, 1)).astype(np.float32)

    # 5) 원본 + tod + dow 순으로 concatenate
    return np.concatenate([raw_data, tod_feat, dow_feat], axis=-1)

def visualize_predictions(model, expanded_data, edge_ids, device, edge_index, edge_attr, interval=(0,480), channel=0, pred_offsets=np.array([3, 6, 12]), window=12):
    """
    모델을 이용해 주어진 edge에 대한 예측값과 실제값을 시각화합니다.

    Parameters
    ----------
    model : torch.nn.Module
        학습된 예측 모델
    data : np.ndarray
        확장된 입력 시계열 데이터 (T, E, C+2)
        dataloader에서 제공하는 방식과 동일하게 add_tod_dow 함수로 확장. 
    edge_ids : list[int]
        시각화할 엣지 인덱스 리스트
    device : torch.device
        모델 연산에 사용할 디바이스
    edge_index : torch.Tensor
        그래프의 엣지 인덱스 정보
    edge_attr : torch.Tensor
        엣지 특성 벡터
    interval : tuple[int, int], optional
        시각화 구간 (start, end)
    channel : int, optional
        예측 대상 채널
    pred_offsets : np.ndarray, optional
        예측 시점 오프셋들
    window : int, optional
        슬라이딩 윈도우 길이
    """
    start, end = interval
    data = expanded_data[start:end]

    model.eval()
    T_total, E, C_all = data.shape

    # 예측값 임시 저장 구조
    pred_lists = {e: defaultdict(list) for e in edge_ids}

    with torch.no_grad():
        for t0 in range(window - 1, T_total - int(pred_offsets.max())):
            x_win = data[t0 - window + 1:t0 + 1, :, :]
            x_tensor = torch.from_numpy(x_win[None]).float().to(device)  # [1, T, E, C_in]
            
            preds = model(x_tensor, edge_index, edge_attr)              # [1, n_pred, E, C_out]
            preds = preds.cpu().numpy()[0]                              # [n_pred, E, C_out]

            for i, offset in enumerate(pred_offsets):
                t_pred = t0 + offset
                if t_pred >= T_total:
                    continue
                for e in edge_ids:
                    # 예: 채널(channel)만 시각화
                    pred_lists[e][t_pred].append(preds[i, e, channel])

    # 평균값으로 시계열 복원
    for e in edge_ids:
        pred_series = np.full(T_total, np.nan, dtype=float)
        for t, vals in pred_lists[e].items():
            pred_series[t] = np.mean(vals)
        actual_series = data[:T_total, e, channel]  # 채널(channel) 실제값

        # 플롯
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(T_total), actual_series,    label=f'Actual Edge {e}')
        plt.plot(np.arange(T_total), pred_series, '--', label=f'Predicted Edge {e}')
        plt.xlabel('Time Step')
        plt.ylabel('Volume Channel')
        plt.title(f'Edge {e}, channel {channel}: Actual vs. Predicted')
        plt.legend()
        plt.tight_layout()
        plt.show()
