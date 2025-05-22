import torch
from tqdm import tqdm

def MAPE(pred, true):
    mask = true.abs() > 1e-3
    return (torch.abs((pred[mask] - true[mask]) / true[mask])).mean().item()

def calculate_loss_and_mape(model, loader, criterion, device, edge_index, edge_attr):
    """
    model : model class
    loader : valid loader (or train loader)
    criterion : criterion
    device : device (cpu or gpu)
    edge_index & edge_attr : from dataset_config
    """
    model.eval()
    
    total_loss = 0.0
    total_mape = 0.0
    n = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, desc="Evaluating"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch, edge_index, edge_attr)  # [B, n_pred, E, C_out]

            # 배치 크기
            B = x_batch.size(0)

            # 손실·MAPE 계산
            loss = criterion(pred, y_batch).item()
            batch_mape = MAPE(pred, y_batch)

            total_loss += loss * B
            total_mape += batch_mape * B
            n += B

    # 전체 평균
    avg_loss = total_loss / n
    avg_mape = total_mape / n

    print(f"Dataset size: {n} samples")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average MAPE: {avg_mape:.4f}")
