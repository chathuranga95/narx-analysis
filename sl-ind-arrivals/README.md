## NARX (PyTorch) for Sri Lanka arrivals

This trains a simple **NARX** model (MLP) following the approach in [Understanding and Using NARX in PyTorch](https://www.codegenes.net/blog/narx-pytorch/).

### Dataset assumptions

`dataset.csv` contains:

- `Month`: date (monthly)
- `Indian_Arrivals`: target \(y(t)\)
- `AVGgoogle_trend`: exogenous input \(u(t)\) (your average of 3 keywords)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train

Default uses 12 lags for both \(y\) and \(u\) (monthly seasonality) and does a time-based split.

```bash
python train_narx.py
```

Artifacts are written to `artifacts/`:

- `narx_model.pt`
- `x_scaler.joblib`
- `y_scaler.joblib`
- `metrics.json`
- `test_predictions.png` (unless `--no-plot`)

### Forecast

After training, you can forecast future arrivals by rolling the model forward:

```bash
python forecast_narx.py --horizon 6
```

Notes:

- Forecasting requires future exogenous inputs. Since `dataset.csv` doesn’t include `AVGgoogle_trend` values beyond the last month, `forecast_narx.py` currently uses **the last observed `AVGgoogle_trend` value** for all forecasted months.
- The output is a table of future `Month` values and `Indian_Arrivals_forecast`.

### Common tweaks

```bash
python train_narx.py --n-y 6 --n-u 6 --hidden 128 --epochs 1000
python train_narx.py --u-cols AVGgoogle_trend --no-plot
```

