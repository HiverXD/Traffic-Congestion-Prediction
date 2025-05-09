# src/utils/Trainer.py

import torch
from torch.cuda.amp import GradScaler, autocast
from IPython.display import clear_output, display
import matplotlib.pyplot as plt

from dataset.dataset_config import edge_index, edge_attr

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        valid_loader,
        optimizer: torch.optim.Optimizer,
        criterion,
        epochs: int = 10,
        device: str = 'cuda',
        print_interval: int = 1,
        plot_interval: int = 1,
        early_stopping_patience: int = 5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        self.print_interval = print_interval
        self.plot_interval = plot_interval
        self.early_stopping_patience = early_stopping_patience

        # edge_index/edge_attr 텐서로 변환 & device 이동
        self.edge_index = torch.from_numpy(edge_index).long().to(device)
        self.edge_attr  = torch.from_numpy(edge_attr).float().to(device)

        # Mixed precision
        self.scaler = GradScaler()

        # 기록
        self.history = {
            'train_loss': [], 'valid_loss': [],
            'train_mape': [], 'valid_mape': []
        }
        self.best_val_loss = float('inf')
        self.no_improve = 0

        # 실시간 플롯 준비
        self.fig, self.ax = plt.subplots()

    @staticmethod
    def mape(pred, true):
        mask = true.abs() > 1e-3
        return (torch.abs((pred[mask] - true[mask]) / true[mask])).mean().item()

    def plot_live(self, epoch):
        self.ax.clear()
        xs = range(1, epoch+1)
        self.ax.plot(xs, self.history['train_loss'], label='Train Loss')
        self.ax.plot(xs, self.history['valid_loss'], label='Valid Loss')
        self.ax.plot(xs, self.history['train_mape'], label='Train MAPE')
        self.ax.plot(xs, self.history['valid_mape'], label='Valid MAPE')
        self.ax.set_xlabel('Epoch')
        self.ax.legend()
        clear_output(wait=True)
        display(self.fig)
        self.fig.canvas.draw()

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            # === Training ===
            self.model.train()
            total_loss = 0.0
            total_mape = 0.0
            n = 0

            for x_batch, y_batch in self.train_loader:
                # (x_batch: [B, T, E, D_in], y_batch: [B, n_pred, E, D_out])
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with autocast():
                    pred = self.model(x_batch, self.edge_index, self.edge_attr)
                    # pred: [B, n_pred, E, D_out]
                    loss = self.criterion(pred, y_batch)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                B = x_batch.size(0)
                total_loss += loss.item() * B
                total_mape += self.mape(pred, y_batch) * B
                n += B

            train_loss = total_loss / n
            train_mape = total_mape / n

            # === Validation ===
            self.model.eval()
            val_loss = 0.0
            val_mape = 0.0
            n_val = 0

            with torch.no_grad():
                for x_batch, y_batch in self.valid_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    pred = self.model(x_batch, self.edge_index, self.edge_attr)
                    loss = self.criterion(pred, y_batch)

                    B = x_batch.size(0)
                    val_loss += loss.item() * B
                    val_mape += self.mape(pred, y_batch) * B
                    n_val += B

            valid_loss = val_loss / n_val
            valid_mape = val_mape / n_val

            # 기록
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            self.history['train_mape'].append(train_mape)
            self.history['valid_mape'].append(valid_mape)

            # 출력 및 플롯
            if epoch % self.print_interval == 0:
                print(f"[Epoch {epoch}] "
                      f"Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}, "
                      f"Train MAPE={train_mape:.4f}, Valid MAPE={valid_mape:.4f}")

            if epoch % self.plot_interval == 0:
                self.plot_live(epoch)

            # Early Stopping
            if valid_loss < self.best_val_loss:
                self.best_val_loss = valid_loss
                self.no_improve = 0
            else:
                self.no_improve += 1
                if self.no_improve >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def get_history(self):
        return self.history
