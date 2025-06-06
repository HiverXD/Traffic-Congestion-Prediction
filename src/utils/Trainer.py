# src/utils/Trainer.py

import torch
from torch.cuda.amp import GradScaler, autocast
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

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
        early_stopping_patience: int = 6,
        auto_save: bool = False,
        save_dir: str = None
    ):
        # 다음에 이어서 학습할 시작 epoch (1-based)
        self.start_epoch = 1
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
        # 자동 저장 기능
        self.auto_save = auto_save
        if save_dir is None:
            save_dir = os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

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
        # 전체 추가할 epoch 수만큼, start_epoch부터 연속으로 진행
        for epoch in tqdm(range(self.start_epoch,
                                 self.start_epoch + self.epochs),
                         desc="Epoch", leave=False):
            # === Training ===
            self.current_epoch = epoch
            self.model.train()
            total_loss = 0.0
            total_mape = 0.0
            n = 0

            # 배치별 진행률 표시
            for x_batch, y_batch in tqdm(
                self.train_loader,
                desc=f"Train {epoch}/{self.epochs}",
                leave=False,
                total=len(self.train_loader)
            ):
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

            # 배치별 진행률 표시
            with torch.no_grad():
                for x_batch, y_batch in tqdm(
                    self.valid_loader,
                    desc=f"Valid {epoch}/{self.epochs}",
                    leave=False,
                    total=len(self.valid_loader)
                ):
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
            if self.print_interval != 0:
                if epoch % self.print_interval == 0:
                    print(f"[Epoch {epoch}] "
                        f"Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}, "
                        f"Train MAPE={train_mape:.4f}, Valid MAPE={valid_mape:.4f}")
                    
            if self.plot_interval != 0:
                if (epoch - self.start_epoch + 1) % self.plot_interval == 0:
                    # plot_live expects total epoch count
                    self.plot_live(epoch)

            # Early Stopping
            if valid_loss < self.best_val_loss:
                self.no_improve = 0

                self.best_val_loss = valid_loss
                # --- Auto-Save on improvement ---
                if self.auto_save:
                    # 모델 이름 + epoch + valid loss 로 파일명 구성
                    model_name = self.model.__class__.__name__
                    filename = f"{model_name}_epoch{epoch:03d}_val{valid_loss:.4f}.pth"
                    path = os.path.join(self.save_dir, filename)
                    torch.save(self.model.state_dict(), path)
                    
            else:
                self.no_improve += 1
                if self.no_improve >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def get_history(self):
        return self.history

    def load_checkpoint(self, state_dict: dict, history: dict = None):
        """
        외부에서 준비된 model.state_dict()와 선택적 history를 받아서
        이어 학습할 준비를 합니다.
        - state_dict: model.state_dict()
        - history: {'train_loss':[...], 'valid_loss':[...], ...}, optional
        """
        # 1) 모델 가중치 로드
        self.model.load_state_dict(state_dict)

        # 2) 과거 기록(history) 복원
        if history is not None:
            self.history = history
            # best_val_loss는 기록된 valid_loss 중 최소로 설정
            self.best_val_loss = min(history.get('valid_loss', [float('inf')]))
            # 시작 epoch = 기존 기록 길이 + 1
            self.start_epoch = len(history.get('train_loss', [])) + 1
        else:
            # 새로운 history로 시작
            self.history = {
                'train_loss': [], 'valid_loss': [],
                'train_mape': [], 'valid_mape': []
            }
            self.best_val_loss = float('inf')
            self.start_epoch = 1

        print(f"[Trainer] Loaded checkpoint. Resuming from epoch {self.start_epoch}.")
    
    def get_best_valid_loss(self):
        return self.best_val_loss
