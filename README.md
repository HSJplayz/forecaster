# forecaster

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![CI](https://github.com/HSJplayz/forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/HSJplayz/forecaster/actions/workflows/ci.yml)
[![Streak Day 19](https://img.shields.io/badge/Streak-Day%2019-green.svg)](https://github.com/HSJplayz/streak)

Classical time-series forecasting **from scratch** — additive decomposition,
simple exponential smoothing, Holt's linear trend, additive Holt-Winters, and
autoregressive (AR) models, plus ACF/PACF, accuracy metrics, and rolling-origin
backtesting. JSON model persistence, a CLI, and a web UI with offline SVG charts.
Only numpy (plus Flask for the web app).

Built with numpy on **Day 19** of the daily projects streak.

## Quick start

```bash
pip install -r requirements.txt

# Decomposition + ACF/PACF diagnostics
python cli.py analyze sales.csv -c value -p 12

# Fit Holt-Winters and forecast 12 steps ahead (writes forecasts + the model)
python cli.py forecast sales.csv -c value -m holt-winters -p 12 --horizon 12 \
    --output forecast.csv --save model.json

# Reuse a saved model without retraining
python cli.py forecast unused.csv --model-file model.json --horizon 12

# Rank all models with rolling-origin backtesting
python cli.py backtest sales.csv -c value -p 12 --horizon 6

# Self-contained demo (no data needed)
python cli.py demo -m holt-winters -p 12 --horizon 12

# Web UI
python app.py   # then open http://127.0.0.1:5000
```

## What it does

| Piece | Details |
| --- | --- |
| **Decomposition** | centered moving average (half-weights for even periods), additive seasonal + residual, seasonal/trend strength |
| **Diagnostics** | `acf`, `pacf` (Durbin-Levinson), plus a non-stationarity hint |
| **Models** | `naive`, `seasonal-naive`, `moving-average`, `ses`, `holt`, additive `holt-winters`, `ar` (AIC order selection) |
| **Metrics** | MAE, RMSE, MAPE, sMAPE, MASE (scaled by the naive error) |
| **Validation** | expanding-window rolling-origin `backtest` and `compare_models` |
| **Persistence** | `save_model`/`load_model` to JSON (state + smoothing params), forecast without refitting |
| **CLI** | `analyze`, `forecast`, `backtest`, `demo` with exit codes 0/1 |
| **Web** | paste CSV → history/fitted/forecast with 95% band, decomposition, ACF/PACF, comparison (offline SVG) |

### Example

```
$ python cli.py backtest sales.csv -c value -p 12 --horizon 6
Backtest 'value': 120 obs, horizon 6, period 12
model                  MAE      RMSE    MAPE%     MASE
ar                   0.557     0.682     1.22    0.154
holt-winters         0.617     0.745     1.35    0.171
seasonal-naive       3.654     3.740     7.90    1.011
naive                4.638     5.546     9.89    1.283
ses                  4.696     5.584    10.01    1.299
moving-average       5.196     5.994    11.06    1.437
holt                 6.971     9.350    14.99    1.929
Best by RMSE: ar
```

## Library use

```python
import numpy as np
from forecast import HoltWinters, compare_models, decompose, make_model

y = ...  # 1-D array of observations

d = decompose(y, period=12)
print(d["seasonal_strength"], d["trend_strength"])

model = make_model("holt-winters", period=12).fit(y, period=12)
print(model.forecast(12))

print(compare_models(y, horizon=6, period=12)["ranking"])
```

## Methods, honestly

- **Smoothing parameters** (α, β, γ) are chosen by a coarse grid search on the
  in-sample one-step SSE — robust and dependency-free, but not a gradient-based
  MLE, so it can land a step away from the optimal value.
- **Holt-Winters** here is the **additive** form and needs at least two full
  seasons; it initializes level/trend/seasonal from the first seasons.
- **AR** orders are picked by minimizing AIC over `1..p_max`, fit with least
  squares on lagged values.
- **Prediction intervals** (CLI/UI) are approximate constant-width bands built
  from the residual σ (`±z·σ`); they do not widen with horizon, so treat them as
  a rough guide, not calibrated intervals.
- **Backtesting** uses an expanding window; every origin refits the model, so
  grid-optimized models are slower to backtest.

## Project layout

```
forecast.py           decomposition, ACF/PACF, models, metrics, backtest, JSON
cli.py                analyze / forecast / backtest / demo commands
app.py                Flask JSON API (/health, /api/forecast)
templates/index.html  self-contained UI with inline SVG charts
test_forecast.py      45 unittest cases
```

## Future scope

- Multiplicative Holt-Winters and damped trend
- SARIMA and automatic order selection (grid + AIC/BIC)
- Calibrated prediction intervals (simulation / analytic)
- Change-point and anomaly detection on residuals
- CSV/Excel input helpers and a downloadable forecast report

## Tests

```bash
python -m unittest test_forecast -v
```

45 tests: decomposition (odd/even windows, seasonal strength), ACF/PACF
behaviour, each model's forecast shape/limits, AR coefficient recovery,
backtest metrics and ranking order, persistence round-trips, CLI exit codes, and
Flask-client API tests. No network.

## License

MIT — see [LICENSE](LICENSE).
