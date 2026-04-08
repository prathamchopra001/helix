"""
LSTM model for time-series anomaly detection.

Architecture:
  Input (batch, 60 days, 29 features)
      ↓
  LSTM (128 hidden, 3 layers, dropout=0.3)
      — reads the sequence day-by-day, builds temporal memory
      ↓
  Take last timestep — the day we're predicting about
      ↓
  Dropout(0.4) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid

Architecture decisions:
  - Temporal attention was tested in v10-v12 but consistently degraded F1.
    With ~9K training windows, the attention layer needs more data to learn
    meaningful timestep weights vs a simple last-step baseline. Reverted to
    proven last-step approach until dataset grows to 50K+ windows.
  - Focal loss is used (NOT weighted BCE). nn.BCELoss(weight=scalar) applies
    the scale factor uniformly to all samples, not just positives — so the
    model defaults to predicting all-zeros with 29 features. Focal loss
    down-weights easy examples and forces learning of anomaly patterns.
    v11 (focal+attention) proved focal loss works: val_f1=0.21 at epoch 1.
  - 29 features (vs original 25): added return_5d, return_20d, hl_range,
    overnight_gap for momentum and volatility signals.
"""
import torch
import torch.nn as nn


class LSTMTransformerModel(nn.Module):
    """
    Name kept as LSTMTransformerModel for MLflow registry compatibility —
    older model versions (1-11) were registered under this class name.
    """
    def __init__(
        self,
        input_size: int = 29,   # 25 base + 4 momentum/volatility features
        hidden_size: int = 128,
        lstm_layers: int = 3,
        lstm_dropout: float = 0.3,
        # kept for API compat with older hyperparameter dicts — unused
        nhead: int = 4,
        transformer_layers: int = 0,
        dim_feedforward: int = 128,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        # Wider output head to match larger hidden_size
        self.output_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, 1) — anomaly probability
        """
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.output_head(last_step)
