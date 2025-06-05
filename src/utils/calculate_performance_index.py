import torch
from tqdm import tqdm

def MAPE(pred, true):
    mask = true.abs() > 1e-3
    return (torch.abs((pred[mask] - true[mask]) / true[mask])).mean().item()

def calculate_performance_index(
    model,
    loader,
    criterion,
    device,
    edge_index,
    edge_attr,
    l2_loss: bool = True,
    mape: bool = True,
    mae: bool = True,
    rmse: bool = True
):
    """
    model      : nn.Module, 예측 모델
    loader     : DataLoader, 평가할 데이터 로더
    criterion  : 손실 함수 (e.g. nn.MSELoss())
    device     : 연산 디바이스 ('cpu' or 'cuda')
    edge_index : 그래프 엣지 인덱스
    edge_attr  : 그래프 엣지 속성
    l2_loss    : L2 손실(MSE) 출력 여부
    mape       : MAPE 출력 여부
    mae        : MAE 출력 여부
    rmse       : RMSE 출력 여부
    """
    model.eval()

    total_loss = 0.0
    total_mape = 0.0
    total_mae  = 0.0
    total_mse  = 0.0
    n_samples  = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, desc="Evaluating"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # 모델 예측
            pred = model(x_batch, edge_index, edge_attr)  # [B, n_pred, E, C_out]
            B = x_batch.size(0)

            # L2 손실 (criterion) 계산
            if l2_loss:
                loss = criterion(pred, y_batch).item()
                total_loss += loss * B

            # MAPE 계산
            if mape:
                batch_mape = MAPE(pred, y_batch)
                total_mape += batch_mape * B

            # MAE 계산
            if mae or rmse:
                batch_mae = torch.mean(torch.abs(pred - y_batch)).item()
                total_mae += batch_mae * B

            # MSE (RMSE 용) 계산
            if rmse:
                batch_mse = torch.mean((pred - y_batch) ** 2).item()
                total_mse += batch_mse * B

            n_samples += B

    # 결과 출력
    print(f"Dataset size: {n_samples} samples")
    if l2_loss:
        avg_loss = total_loss / n_samples
        print(f"Average Loss (L2/MSE): {avg_loss:.4f}")
    if mape:
        avg_mape = total_mape / n_samples
        print(f"Average MAPE:         {avg_mape:.4f}")
    if mae:
        avg_mae = total_mae / n_samples
        print(f"Average MAE:          {avg_mae:.4f}")
    if rmse:
        avg_rmse = (total_mse / n_samples) ** 0.5
        print(f"Average RMSE:         {avg_rmse:.4f}")
