# 필요한 라이브러리
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
        epochs: int = 10,
        device: str = 'cuda',
        print_interval: int = 1,
        plot_interval: int = 1,
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

        # 기록 저장
        self.history = {
            'train_loss': [], 'valid_loss': [],
            'train_mape': [], 'valid_mape': []
        }
        self.best_val_loss = float('inf')
        self.no_improve = 0
        self.scaler = GradScaler()

        # 실시간 플롯 준비
        self.fig, self.ax = plt.subplots()

    @staticmethod
    def mape(pred, true):
        mask = true.abs() > 1e-3
        return (torch.abs((pred[mask] - true[mask]) / true[mask])).mean().item()

    def _unpack_batch(self, batch):
        """
        배치에서 모델 입력(input)과 타깃(target)을 분리하는 메서드.
        배치 형태에 맞게 오버라이드하거나 조건문을 추가하세요.
        """
        # 예시: dict 형태
        if isinstance(batch, dict):
            return batch['past_edges'], batch['future_edges']
        # PyG Data or Batch
        if isinstance(batch, (Data, Batch)):
            # x, y 프로퍼티에 담겨 있다고 가정
            return batch.x, batch.y
        raise ValueError("지원하지 않는 배치 형식입니다.")

    def _predict(self, inputs):
        """
        모델 호출 방식 추상화.
        단일 입력, 다중 입력 등 모델 시그니처에 맞게 수정하세요.
        """
        # 예시: 단일 텐서 입력
        return self.model(inputs)
        # 또는
        # return self.model(inputs.x, inputs.edge_index, inputs.edge_attr)

    def _move_to_device(self, tensor):
        """
        텐서나 Data/Batch 객체를 device로 옮기는 유틸.
        """
        if isinstance(tensor, (Data, Batch)):
            return tensor.to(self.device)
        return tensor.to(self.device)

    def plot_live(self, epoch):
        """주기적 실시간 플롯."""
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
            total_loss, total_mape, n = 0.0, 0.0, 0
            for batch in self.train_loader:
                inputs, targets = self._unpack_batch(batch)
                inputs, targets = self._move_to_device(inputs), self._move_to_device(targets)
                B = targets.size(0)

                with autocast():
                    pred = self._predict(inputs)
                    loss = self.criterion(pred, targets)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item() * B
                total_mape += self.mape(pred, targets) * B
                n += B

            train_loss = total_loss / n
            train_mape = total_mape / n

            # === Validation ===
            self.model.eval()
            val_loss, val_mape, n_val = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in self.valid_loader:
                    inputs, targets = self._unpack_batch(batch)
                    inputs, targets = self._move_to_device(inputs), self._move_to_device(targets)
                    B = targets.size(0)

                    pred = self._predict(inputs)
                    loss = self.criterion(pred, targets)

                    val_loss += loss.item() * B
                    val_mape += self.mape(pred, targets) * B
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
                if self.no_improve >= 5:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def get_history(self):
        return self.history
