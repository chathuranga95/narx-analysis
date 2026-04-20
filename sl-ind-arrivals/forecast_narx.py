import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    x_scaler: object
    y_scaler: object
    n_y: int
    n_u: int
    y_col: str
    u_cols: List[str]


class NARX(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_bundle(artifacts_dir: Path) -> ModelBundle:
    ckpt = torch.load(artifacts_dir / "narx_model.pt", map_location="cpu")
    model = NARX(input_size=int(ckpt["input_size"]), hidden_size=int(ckpt["hidden_size"]), output_size=1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_scaler = joblib.load(artifacts_dir / "x_scaler.joblib")
    y_scaler = joblib.load(artifacts_dir / "y_scaler.joblib")
    return ModelBundle(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        n_y=int(ckpt["n_y"]),
        n_u=int(ckpt["n_u"]),
        y_col=str(ckpt["y_col"]),
        u_cols=list(ckpt["u_cols"]),
    )


def month_add(start: pd.Timestamp, k: int) -> pd.Timestamp:
    # Monthly frequency add (keeps day=1 if present)
    return (start.to_period("M") + k).to_timestamp()


def main() -> None:
    p = argparse.ArgumentParser(description="Forecast future months with trained NARX artifacts")
    p.add_argument("--csv", type=str, default="dataset.csv")
    p.add_argument("--date-col", type=str, default="Month")
    p.add_argument("--artifacts", type=str, default="artifacts")
    p.add_argument("--horizon", type=int, default=6, help="number of months to forecast")
    p.add_argument(
        "--u-future",
        type=str,
        default="last",
        choices=["last"],
        help="strategy for future exogenous values (only 'last' supported)",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.sort_values(args.date_col).reset_index(drop=True)

    bundle = load_bundle(Path(args.artifacts))

    for col in [bundle.y_col] + bundle.u_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV")
    df = df.dropna(subset=[bundle.y_col] + bundle.u_cols).reset_index(drop=True)

    y_hist = df[[bundle.y_col]].to_numpy(dtype=np.float32).reshape(-1)  # (T,)
    u_hist = df[bundle.u_cols].to_numpy(dtype=np.float32)  # (T, n_exog)

    if len(y_hist) < bundle.n_y + 1 or len(u_hist) < bundle.n_u + 1:
        raise ValueError("Not enough history for configured lags")

    last_month = pd.Timestamp(df[args.date_col].iloc[-1])
    last_u = u_hist[-1].copy()

    y_window = list(y_hist[-bundle.n_y :].tolist())
    u_window = u_hist[-bundle.n_u :].copy()  # (n_u, n_exog)

    preds = []
    for step in range(1, args.horizon + 1):
        if args.u_future == "last":
            u_next = last_u
        else:
            raise ValueError("Unsupported u-future strategy")

        x_vec = np.concatenate([np.array(y_window, dtype=np.float32), u_window.reshape(-1).astype(np.float32)], axis=0)
        x_s = bundle.x_scaler.transform(x_vec.reshape(1, -1))
        with torch.no_grad():
            y_pred_s = bundle.model(torch.from_numpy(x_s).float()).numpy()
        y_pred = float(bundle.y_scaler.inverse_transform(y_pred_s)[0, 0])

        forecast_month = month_add(last_month, step)
        preds.append((forecast_month.strftime("%Y-%m"), y_pred))

        # roll windows forward
        y_window = y_window[1:] + [y_pred]
        u_window = np.vstack([u_window[1:], u_next.reshape(1, -1)])

    out = pd.DataFrame(preds, columns=[args.date_col, f"{bundle.y_col}_forecast"])
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

