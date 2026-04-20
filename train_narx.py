import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class Split:
    x: torch.Tensor
    y: torch.Tensor


class NARX(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_supervised(
    y: np.ndarray, u: np.ndarray, n_y: int, n_u: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build NARX supervised data:
      y(t) = f(y(t-1..t-n_y), u(t-1..t-n_u))
    """
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("y must be shape (T, 1)")
    if u.ndim != 2:
        raise ValueError("u must be shape (T, n_exog)")
    if len(y) != len(u):
        raise ValueError("y and u must have same length")
    if n_y < 1 or n_u < 1:
        raise ValueError("n_y and n_u must be >= 1")

    t0 = max(n_y, n_u)
    xs = []
    ys = []
    for t in range(t0, len(y)):
        past_y = y[t - n_y : t].reshape(-1)  # (n_y,)
        past_u = u[t - n_u : t].reshape(-1)  # (n_u * n_exog,)
        xs.append(np.concatenate([past_y, past_u], axis=0))
        ys.append(y[t])

    x = np.stack(xs, axis=0).astype(np.float32)
    y_out = np.stack(ys, axis=0).astype(np.float32)
    return x, y_out


def time_split(
    x: np.ndarray, y: np.ndarray, train_frac: float, val_frac: float
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be between 0 and 1")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be between 0 and 1")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")

    n = len(x)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
    x_test, y_test = x[n_train + n_val :], y[n_train + n_val :]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple NARX MLP on dataset.csv")
    parser.add_argument("--csv", type=str, default="dataset.csv")
    parser.add_argument("--date-col", type=str, default="Month")
    parser.add_argument("--y-col", type=str, default="Indian_Arrivals")
    parser.add_argument("--u-cols", type=str, default="AVGgoogle_trend")
    parser.add_argument("--n-y", type=int, default=12, help="number of output lags")
    parser.add_argument("--n-u", type=int, default=12, help="number of input lags")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
        df = df.sort_values(args.date_col)

    u_cols = [c.strip() for c in args.u_cols.split(",") if c.strip()]
    keep_cols = [args.y_col] + u_cols
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")

    df = df.dropna(subset=keep_cols).reset_index(drop=True)

    y_raw = df[[args.y_col]].to_numpy(dtype=np.float32)
    u_raw = df[u_cols].to_numpy(dtype=np.float32)

    x_raw, y_sup = make_supervised(y_raw, u_raw, n_y=args.n_y, n_u=args.n_u)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = time_split(
        x_raw, y_sup, train_frac=args.train_frac, val_frac=args.val_frac
    )

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_train_s = x_scaler.fit_transform(x_train)
    y_train_s = y_scaler.fit_transform(y_train)
    x_val_s = x_scaler.transform(x_val)
    y_val_s = y_scaler.transform(y_val)
    x_test_s = x_scaler.transform(x_test)
    y_test_s = y_scaler.transform(y_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NARX(input_size=x_train_s.shape[1], hidden_size=args.hidden, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def to_split(xn: np.ndarray, yn: np.ndarray) -> Split:
        return Split(
            x=torch.from_numpy(xn).float().to(device),
            y=torch.from_numpy(yn).float().to(device),
        )

    train = to_split(x_train_s, y_train_s)
    val = to_split(x_val_s, y_val_s)
    test = to_split(x_test_s, y_test_s)

    best_val = float("inf")
    best_state = None
    patience_left = args.patience

    n_train = train.x.shape[0]
    batch_size = max(1, min(args.batch_size, n_train))

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = 0.0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = train.x[idx]
            yb = train.y[idx]

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(idx)

        train_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(val.x)
            val_loss = float(criterion(val_pred, val.y).item())

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch == 1 or epoch % 25 == 0 or patience_left == 0:
            print(f"Epoch {epoch:4d} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | best_val={best_val:.6f}")

        if patience_left == 0:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best_state")
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred_s = model(test.x).cpu().numpy()

    test_pred = y_scaler.inverse_transform(test_pred_s)
    test_true = y_test

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "narx_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": x_train_s.shape[1],
            "hidden_size": args.hidden,
            "n_y": args.n_y,
            "n_u": args.n_u,
            "y_col": args.y_col,
            "u_cols": u_cols,
        },
        model_path,
    )
    joblib.dump(x_scaler, out_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, out_dir / "y_scaler.joblib")

    metrics = {
        "test_rmse": rmse(test_pred.reshape(-1), test_true.reshape(-1)),
        "test_mae": mae(test_pred.reshape(-1), test_true.reshape(-1)),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "device": str(device),
    }
    (out_dir / "metrics.json").write_text(pd.Series(metrics).to_json(indent=2), encoding="utf-8")

    print("Saved artifacts to:", out_dir.resolve())
    print("Test RMSE:", metrics["test_rmse"])
    print("Test MAE:", metrics["test_mae"])

    if not args.no_plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(11, 4))
        plt.plot(test_true.reshape(-1), label="Actual")
        plt.plot(test_pred.reshape(-1), label="Predicted")
        plt.title("NARX test set: actual vs predicted")
        plt.xlabel("Test time index")
        plt.ylabel(args.y_col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "test_predictions.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
