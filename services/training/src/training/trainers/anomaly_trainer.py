"""
Training loop for the LSTM + Temporal Attention anomaly detection model.

Key decisions explained:

Focal Loss (vs weighted BCELoss):
  nn.BCELoss(weight=scalar) applies the same scale factor to ALL samples,
  not just positives — it doesn't implement class-weighted BCE. The model
  defaults to predicting all-zeros (majority class) unless the loss function
  specifically up-weights anomaly samples. Focal Loss does this via the
  (1-p)^gamma modulation: easy normal days contribute exponentially less
  gradient, forcing the model to learn anomaly patterns. gamma=2 is standard
  (RetinaNet, Lin et al. 2017). Note: focal loss + attention (v11) got F1=0.28.
  v12-v13 tested BCE + no attention and got F1=0.16/0.05 — confirming that
  BCE doesn't work with 29 features regardless of attention.

Gradient clipping:
  LSTMs are prone to "exploding gradients" where a single large gradient update
  destabilizes training. Clipping at 1.0 caps the gradient norm — training stays
  stable without slowing convergence.

Early stopping:
  We track F1 score on the validation set after each epoch. If it doesn't
  improve for `patience` epochs, training stops. This prevents overfitting —
  the model stops learning the training data's noise and starts generalizing.
  We always restore the best checkpoint, not the last one.

MLflow logging:
  Every run is tracked: hyperparameters, per-epoch metrics, final model artifact.
  The model is registered in MLflow Model Registry as a new version in "Staging".
  The retraining DAG promotes it to "Production" only if F1 improves by > 0.02.
"""

import os
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from shared.logging import get_logger
from training.datasets.time_series_dataset import AnomalyWindowDataset
from training.models.lstm_transformer import LSTMTransformerModel

log = get_logger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "anomaly_detection"
MODEL_NAME = "helix_anomaly_detector"


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma: focusing parameter — higher values down-weight easy examples more.
           gamma=0 reduces to weighted BCE. gamma=2 is the standard from
           Lin et al. (RetinaNet, 2017) and works well in practice.
    alpha: per-class weight derived from class imbalance ratio.
           alpha = pos_weight / (1 + pos_weight) for the positive class.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = pos_weight / (1.0 + pos_weight)  # weight for positive class

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = functional.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)  # probability of correct class
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


def _compute_pos_weight(y_train: np.ndarray) -> float:
    """Ratio of normal to anomaly samples — used to weight the loss."""
    n_anomaly = y_train.sum()
    n_normal = len(y_train) - n_anomaly
    if n_anomaly == 0:
        return 1.0
    return float(n_normal / n_anomaly)


def _find_best_threshold(
    model: "LSTMTransformerModel",
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float] | None = None,
) -> tuple[float, float]:
    """
    Sweep classification thresholds on the validation set and return
    (best_threshold, best_f1) — the threshold that maximises F1 score.

    Why not always use 0.5?
      With class imbalance (10-17% anomalies) the model's raw output
      distribution is shifted. Thresholds in the range 0.3-0.45 often
      give higher recall without sacrificing too much precision.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.25, 0.75, 0.05)]

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = model(X_batch.to(device)).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_batch.numpy().astype(int).tolist())

    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)

    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        preds = (all_probs_arr >= t).astype(int)
        f1 = f1_score(all_labels_arr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    return best_thresh, best_f1


def _evaluate(
    model: LSTMTransformerModel,
    loader: DataLoader,
    criterion: "nn.Module",
    device: torch.device,
) -> dict[str, float]:
    """Run model on a DataLoader and return loss + classification metrics."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch).squeeze(1)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(X_batch)
            all_preds.extend((preds.cpu().numpy() > 0.5).astype(int))
            all_labels.extend(y_batch.cpu().numpy().astype(int))

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    return {"loss": avg_loss, "f1": f1, "precision": precision, "recall": recall}


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hyperparams: dict[str, Any] | None = None,
    correlation_id: str = "",
) -> dict[str, Any]:
    """
    Train the model and log everything to MLflow.

    Returns a dict with mlflow_run_id, model_version, and best val F1.
    """
    # Infer input_size dynamically from data — handles feature count changes
    # without needing to update this file when new features are added.
    n_features = X_train.shape[2]

    hp = {
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 150,
        "patience": 20,
        "hidden_size": 128,
        "lstm_layers": 3,
        "lstm_dropout": 0.3,
        "nhead": 4,
        "transformer_layers": 0,
        "dim_feedforward": 128,
        "grad_clip": 1.0,
        "input_size": n_features,
        **(hyperparams or {}),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("training_start", device=str(device), correlation_id=correlation_id)

    train_ds = AnomalyWindowDataset(X_train, y_train)
    val_ds = AnomalyWindowDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp["batch_size"], shuffle=False)

    model = LSTMTransformerModel(
        input_size=hp["input_size"],
        hidden_size=hp["hidden_size"],
        lstm_layers=hp["lstm_layers"],
        lstm_dropout=hp["lstm_dropout"],
        nhead=hp["nhead"],
        transformer_layers=hp["transformer_layers"],
        dim_feedforward=hp["dim_feedforward"],
    ).to(device)

    pos_weight_val = _compute_pos_weight(y_train)
    # FocalLoss: gamma=2 down-weights easy normal-day examples so the model
    # focuses gradient budget on hard anomaly examples.
    # Note: nn.BCELoss(weight=scalar) scales ALL samples uniformly — it doesn't
    # do class-weighted BCE — so focal loss is used instead. v11 (focal+attention)
    # proved focal loss starts learning immediately (val_f1=0.21 at epoch 1 vs
    # 0.00 for uniform-weight BCE). Attention was the v11 culprit, not focal loss.
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    # Halve LR when val F1 plateaus for 5 epochs — helps escape local minima
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_val_f1 = 0.0
    best_state: dict = {}
    patience_counter = 0

    with mlflow.start_run() as run:
        mlflow.log_params(hp)
        mlflow.log_param("pos_weight", round(pos_weight_val, 3))
        mlflow.log_param("train_windows", len(X_train))
        mlflow.log_param("val_windows", len(X_val))

        for epoch in range(1, hp["epochs"] + 1):
            # Training pass
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                preds = model(X_batch).squeeze(1)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_loader.dataset)
            val_metrics = _evaluate(model, val_loader, criterion, device)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                },
                step=epoch,
            )

            log.info(
                "epoch_complete",
                epoch=epoch,
                train_loss=round(train_loss, 4),
                val_f1=round(val_metrics["f1"], 4),
                correlation_id=correlation_id,
            )

            scheduler.step(val_metrics["f1"])

            # Early stopping
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= hp["patience"]:
                    log.info("early_stopping", epoch=epoch, best_val_f1=round(best_val_f1, 4))
                    break

        # Restore best weights and log to MLflow
        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_f1", best_val_f1)

        # Find the threshold that maximises F1 on the validation set
        best_thresh, thresh_f1 = _find_best_threshold(model, val_loader, device)
        model.threshold = best_thresh  # stored as model attribute — saved with the artifact
        mlflow.log_param("classification_threshold", best_thresh)
        mlflow.log_metric("threshold_val_f1", thresh_f1)
        log.info(
            "threshold_optimised",
            threshold=best_thresh,
            val_f1_at_threshold=round(thresh_f1, 4),
            correlation_id=correlation_id,
        )

        mlflow.pytorch.log_model(model, artifact_path="model")

        # Register separately to get the version number
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        model_version = mv.version

        log.info(
            "training_complete",
            best_val_f1=round(best_val_f1, 4),
            model_version=model_version,
            run_id=run.info.run_id,
            correlation_id=correlation_id,
        )

        return {
            "mlflow_run_id": run.info.run_id,
            "model_version": str(model_version),
            "best_val_f1": best_val_f1,
        }
