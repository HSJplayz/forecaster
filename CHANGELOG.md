# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-10-20

### Added
- `forecast.py`: classical time-series analysis and forecasting, numpy only
  - Additive decomposition (`centered_moving_average`, `decompose`) with
    seasonal/trend strength and residual std.
  - `acf` and `pacf` (Durbin-Levinson recursion).
  - Models: `Naive`, `SeasonalNaive`, `MovingAverage`,
    `SimpleExponentialSmoothing`, `Holt`, additive `HoltWinters`, and `AR`
    (order chosen by AIC). Smoothing parameters are grid-optimized on in-sample
    SSE.
  - Metrics (`mae`, `rmse`, `mape`, `smape`, `mase`) and expanding-window
    rolling-origin `backtest` + `compare_models`.
  - JSON `save_model`/`load_model` with a `_model` registry dispatch.
- `cli.py`: `analyze`, `forecast` (with `--output`/`--save`/`--model-file`),
  `backtest` (auto comparison), and `demo`; exit codes 0/1.
- `app.py` + `templates/index.html`: Flask `/health` and `POST /api/forecast`
  returning chart-ready JSON, with a self-contained UI (history + fitted +
  forecast with 95% band, decomposition, ACF/PACF, comparison table) drawn as
  inline SVG.
- `test_forecast.py`: 45 unittest cases covering decomposition, ACF/PACF,
  every model, metrics, backtesting, persistence round-trips, CLI exit codes,
  and the Flask API. No network.

[1.0.0]: https://github.com/HSJplayz/forecaster/releases/tag/v1.0.0
