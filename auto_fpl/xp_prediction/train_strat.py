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
import matplotlib.pyplot as plt
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

def _balance_train_by_bin(X, y, bins, strategy="undersample", random_state=42, target_per_bin=None):
    """
    Return balanced (X, y, bins) for training only.
    - strategy: "undersample" (default) or "oversample"
    - target_per_bin:
        * None  -> min count across bins (for undersample) or max count (for oversample)
        * int   -> explicit per-bin size
    """
    rng = np.random.default_rng(random_state)
    # make a single indexable table
    df = pd.DataFrame(X)
    df["__y__"] = y
    df["__bin__"] = bins

    counts = df["__bin__"].value_counts()
    if strategy == "undersample":
        k = target_per_bin if target_per_bin is not None else counts.min()
        take = lambda g: g.sample(n=min(k, len(g)), random_state=random_state)
    elif strategy == "oversample":
        k = target_per_bin if target_per_bin is not None else counts.max()
        def take(g):
            if len(g) >= k:
                return g.sample(n=k, random_state=random_state)
            # sample with replacement
            idx = rng.choice(g.index.values, size=k, replace=True)
            return g.loc[idx]
    else:
        raise ValueError("strategy must be 'undersample' or 'oversample'")

    balanced = df.groupby("__bin__", group_keys=False).apply(take).sample(frac=1.0, random_state=random_state)
    y_bal = balanced.pop("__y__").to_numpy()
    bins_bal = balanced.pop("__bin__").to_numpy()
    X_bal = balanced.to_numpy()
    return X_bal, y_bal, bins_bal


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

def _save_parity_plot(y_true, y_pred, out_path: str, title: str = "Parity plot"):
    import numpy as np
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # sensible limits (pad a bit)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad

    plt.figure(figsize=(6, 6))
    # scatter (downsample if huge)
    if len(y_true) > 100_000:
        idx = np.random.choice(len(y_true), size=100_000, replace=False)
        plt.scatter(y_true[idx], y_pred[idx], s=2, alpha=0.4)
    elif len(y_true) > 20_000:
        # hexbin helps when dense
        plt.hexbin(y_true, y_pred, gridsize=60, mincnt=1)
        cb = plt.colorbar()
        cb.set_label("count")
    else:
        plt.scatter(y_true, y_pred, s=6, alpha=0.6)

    # 45° line
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual points")
    plt.ylabel("Predicted points")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def train_xp_model(
    data_path: str,
    output_root: str = "../models",            # <-- parent dir where unique run dirs are created
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 8,
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
    bins = pd.cut(y, bins=[-np.inf, 3, 9, np.inf], labels=False)
    X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
        X, y, bins, test_size=test_size, random_state=random_state, stratify=bins
    )

    # ------------- balance TRAIN ONLY -------------
    # choose one:
    #   a) undersample to the smallest bin (safer, avoids overfitting)
    X_train, y_train, bins_train = _balance_train_by_bin(
        X_train, y_train, bins_train, strategy="undersample", random_state=random_state
    )
    #   b) OR oversample to the largest bin (comment the block above and use this instead)
    # X_train, y_train, bins_train = _balance_train_by_bin(
    #     X_train, y_train, bins_train, strategy="oversample", random_state=random_state
    # )

    # ------------- scale -------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print(len(X_train))
    X_test  = scaler.transform(X_test)

    # ------------- datasets / loaders -------------
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


    # ------------- model / opt / loss -------------
    model = XPModel(input_size=X_train.shape[1], hidden_size=hidden_size)
    criterion = torch.nn.HuberLoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True, min_lr=1e-6
    )

    # ------------- training loop -------------
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []  # collect per-epoch train/val losses
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb).mean()
            train_losses.append(float(loss.detach().cpu()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # validation on the test split (held-out)
        model.eval()
        val_losses = []
        y_val_true, y_val_pred = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                outputs = model(inputs).squeeze()
                loss_vec = criterion(outputs, targets)   # elementwise SmoothL1
                val_losses.append(loss_vec.mean().item())
                y_val_true.append(targets.cpu().numpy())
                y_val_pred.append(outputs.cpu().numpy())

        y_val_true = np.concatenate(y_val_true)
        y_val_pred = np.concatenate(y_val_pred)

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_mse = mean_squared_error(y_val_true, y_val_pred)
        scheduler.step(val_mse)

        print(f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MSE: {val_mse:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_mse": val_mse,
        })

        # --- save best model by lowest MSE ---
        if val_mse < best_val_loss:  # rename var to best_val_mse if you want clarity
            best_val_loss = val_mse
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

    # --- save parity plots ---
    _save_parity_plot(train_targets, train_preds, os.path.join(run_dir, "parity_train.png"), "Train parity")
    _save_parity_plot(test_targets,  test_preds,  os.path.join(run_dir, "parity_test.png"),  "Test parity")

    # (optional) save raw predictions too
    pd.DataFrame({
        "split": ["train"] * len(train_targets) + ["test"] * len(test_targets),
        "y_true": np.concatenate([train_targets, test_targets]),
        "y_pred": np.concatenate([train_preds,  test_preds]),
    }).to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    # ------------- save config (useful for reproducibility) -------------
    config: Dict[str, Any] = {
        "data_path": data_path,
        "output_root": output_root,
        "split": "strat",
        "run_id": run_id,
        "model_class": "XPModel",
        "input_size": X_train.shape[1],
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "test_size": test_size,
        "random_state": random_state,
        "best_val_loss": round(float(best_val_loss), 2),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll artifacts saved to: {run_dir}")
    return run_dir

if __name__ == "__main__":
    train_xp_model("../data/incl_set_pieces.csv", hidden_size=[64, 64], learning_rate=1e-4)
