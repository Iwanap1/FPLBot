# train.py
import os
import uuid
import json
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

from model import XPModel


def _timestamp_uid() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    short = str(uuid.uuid4())[:8]
    return f"{ts}-{short}"


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
    }


def train_xp_model(
    data_path: str,
    output_root: str = "../models",            # <-- parent dir where unique run dirs are created
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 5,
    learning_rate: float = 1e-4,
    test_size: float = 0.2,
    random_state: int = 42,
    hidden_size: int = 64
) -> str:
    """
    Trains XPModel and saves all artifacts in a unique run directory under `output_root`.

    Returns:
        run_dir (str): path to the unique directory containing all artifacts.
    """
    # ------------- make unique run dir -------------
    run_id = _timestamp_uid()
    run_dir = _ensure_dir(os.path.join(output_root, run_id))

    # ------------- load data -------------
    data = pd.read_csv(data_path)
    feature_columns = [col for col in data.columns if col not in ["player_id", "game_week", "season", "points"]]
    X = data[feature_columns].values
    y = data["points"].values

    # ------------- split -------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ------------- scale -------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------- datasets / loaders -------------
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------------- model / opt / loss -------------
    model = XPModel(input_size=X_train.shape[1], hidden_size=hidden_size)
    criterion = torch.nn.HuberLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------- training loop -------------
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []  # collect per-epoch train/val losses
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # validation on the test split (held-out)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # early stopping + checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    # ------------- save artifacts -------------
    # scaler + features
    joblib.dump(scaler, os.path.join(run_dir, "scaler.pkl"))
    with open(os.path.join(run_dir, "features.json"), "w") as f:
        json.dump(feature_columns, f)

    # training log
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "training_log.csv"), index=False)

    # ------------- compute & save final metrics (train and test) -------------
    model.eval()
    with torch.no_grad():
        # train preds
        train_preds = []
        train_targets = []
        for xb, yb in DataLoader(train_dataset, batch_size=batch_size, shuffle=False):
            train_preds.append(model(xb).squeeze().cpu().numpy())
            train_targets.append(yb.cpu().numpy())
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)

        # test preds
        test_preds = []
        test_targets = []
        for xb, yb in DataLoader(test_dataset, batch_size=batch_size, shuffle=False):
            test_preds.append(model(xb).squeeze().cpu().numpy())
            test_targets.append(yb.cpu().numpy())
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)

    train_metrics = _compute_metrics(train_targets, train_preds)
    test_metrics = _compute_metrics(test_targets, test_preds)

    metrics_rows = [
        {"split": "train", **train_metrics},
        {"split": "test", **test_metrics},
    ]
    pd.DataFrame(metrics_rows).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # ------------- save config (useful for reproducibility) -------------
    config: Dict[str, Any] = {
        "data_path": data_path,
        "output_root": output_root,
        "run_id": run_id,
        "model_class": "XPModel",
        "input_size": X_train.shape[1],
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "training": "Huber normal",
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "test_size": test_size,
        "random_state": random_state,
        "best_val_loss": best_val_loss,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll artifacts saved to: {run_dir}")
    return run_dir

if __name__ == "__main__":
    train_xp_model("../data/incl_set_pieces.csv", hidden_size=[32,32])
