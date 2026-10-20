"""From-scratch time-series analysis and forecasting on numpy only.

Includes classical additive decomposition, ordinary/seasonal naive, moving
average, simple exponential smoothing, Holt's linear trend, additive
Holt-Winters, and autoregressive (AR) models, plus ACF/PACF, accuracy metrics,
rolling-origin backtesting, and JSON persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EPS = 1e-12


def _as_1d(y) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 2:
        raise ValueError("series must contain at least 2 observations")
    if np.isnan(y).any():
        raise ValueError("series contains NaN values")
    return y


# --- decomposition ------------------------------------------------------------

def centered_moving_average(y, window: int) -> np.ndarray:
    """Centered moving average; even windows use half-weights at the ends."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.full(n, np.nan)
    if window < 1 or window > n:
        return out
    if window % 2 == 1:
        half = (window - 1) // 2
        for t in range(half, n - half):
            out[t] = y[t - half:t + half + 1].mean()
    else:
        half = window // 2
        for t in range(half, n - half):
            out[t] = (0.5 * y[t - half] + y[t - half + 1:t + half].sum()
                      + 0.5 * y[t + half]) / window
    return out


def decompose(y, period: int) -> dict:
    """Classical additive decomposition: y = trend + seasonal + residual."""
    y = _as_1d(y)
    n = len(y)
    if period < 2:
        raise ValueError("period must be >= 2")
    if n < 2 * period:
        raise ValueError(f"need at least {2 * period} observations for period={period}")
    trend = centered_moving_average(y, period)
    detrended = y - trend

    # average detrended values per phase, then center the seasonal component
    seasonal_means = np.full(period, np.nan)
    for p in range(period):
        vals = detrended[p::period]
        vals = vals[~np.isnan(vals)]
        if vals.size:
            seasonal_means[p] = vals.mean()
    seasonal_means = np.nan_to_num(seasonal_means, nan=0.0)
    seasonal_means -= seasonal_means.mean()
    seasonal = np.array([seasonal_means[t % period] for t in range(n)])
    resid = y - trend - seasonal

    mask = ~np.isnan(trend)
    var_resid = np.nanvar(resid)
    var_detrended = np.nanvar(detrended)
    var_deseason = np.nanvar(y - seasonal)
    strength_trend = max(0.0, 1.0 - var_resid / var_deseason) if var_deseason > EPS else 0.0
    strength_seasonal = max(0.0, 1.0 - var_resid / var_detrended) if var_detrended > EPS else 0.0

    return {
        "observed": y,
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid,
        "period": period,
        "seasonal_strength": float(strength_seasonal),
        "trend_strength": float(strength_trend),
        "resid_std": float(np.nanstd(resid[mask])) if mask.any() else float(np.nanstd(resid)),
    }


# --- autocorrelation ----------------------------------------------------------

def acf(y, nlags: int) -> np.ndarray:
    y = _as_1d(y)
    n = len(y)
    nlags = min(nlags, n - 1)
    c = y - y.mean()
    denom = float((c * c).sum())
    out = [1.0]
    for k in range(1, nlags + 1):
        out.append(float((c[:n - k] * c[k:]).sum() / denom) if denom > EPS else 0.0)
    return np.asarray(out)


def pacf(y, nlags: int) -> np.ndarray:
    """Partial autocorrelation via the Durbin-Levinson recursion."""
    y = _as_1d(y)
    nlags = min(nlags, len(y) - 1)
    r = acf(y, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    pac = np.zeros(nlags + 1)
    if nlags >= 1:
        phi[1, 1] = r[1]
        pac[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = num / den if abs(den) > 1e-9 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pac[k] = phi[k, k]
    return pac


# --- metrics ------------------------------------------------------------------

def _metrics(actual, forecast, *, naive_scale=None) -> dict:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    err = actual - forecast
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(actual) < EPS, np.nan, np.abs(actual))
    mape = float(np.nanmean(np.abs(err) / denom) * 100) if np.any(~np.isnan(denom)) else None
    smape = float(np.mean(2 * np.abs(err) / (np.abs(actual) + np.abs(forecast) + EPS)) * 100)
    mase = float(mae / naive_scale) if naive_scale and naive_scale > EPS else None
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "mase": mase}


def naive_scale(y, period: int = 1) -> float:
    """Denominator for MASE: mean absolute in-sample naive (or seasonal naive) error."""
    y = np.asarray(y, dtype=float)
    if period <= 1:
        diffs = np.abs(np.diff(y))
    else:
        diffs = np.abs(y[period:] - y[:-period])
    return float(diffs.mean()) if diffs.size else float(np.abs(y).mean())


# --- smoothing helpers --------------------------------------------------------

def _ses_sse(y, alpha):
    s = y[0]
    sse = 0.0
    for t in range(1, len(y)):
        sse += (y[t] - s) ** 2
        s = alpha * y[t] + (1 - alpha) * s
    return sse


def _holt_sse(y, a, b):
    level, trend = y[0], y[1] - y[0]
    sse = 0.0
    for t in range(1, len(y)):
        sse += (y[t] - (level + trend)) ** 2
        new_level = a * y[t] + (1 - a) * (level + trend)
        new_trend = b * (new_level - level) + (1 - b) * trend
        level, trend = new_level, new_trend
    return sse, level, trend


def _hw_sse(y, m, a, b, g):
    n = len(y)
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m if n >= 2 * m else 0.0
    seasonal = y[:m] - level
    seasonal -= seasonal.mean()
    sse = 0.0
    for t in range(m, n):
        f = level + trend + seasonal[t % m]
        sse += (y[t] - f) ** 2
        new_level = a * (y[t] - seasonal[t % m]) + (1 - a) * (level + trend)
        new_trend = b * (new_level - level) + (1 - b) * trend
        seasonal[t % m] = g * (y[t] - new_level) + (1 - g) * seasonal[t % m]
        level, trend = new_level, new_trend
    return sse, level, trend, seasonal


_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _best_ses(y):
    best = min(((_ses_sse(y, a), a) for a in _GRID))
    return best[1]


def _best_holt(y):
    best = min(((*_holt_sse(y, a, b)[:1], a, b) for a in _GRID for b in _GRID))
    return best[1], best[2]


def _best_hw(y, m):
    best = min((_hw_sse(y, m, a, b, g)[0], a, b, g)
               for a in _GRID for b in _GRID for g in _GRID)
    return best[1], best[2], best[3]


# --- models -------------------------------------------------------------------

class BaseForecaster:
    name = "base"

    def fit(self, y, *, period: int = 1):
        self.y = _as_1d(y)
        self.period = period
        self.fitted_ = np.full(len(self.y), np.nan)
        return self

    def forecast(self, h: int) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_params(cls, p: dict):
        raise NotImplementedError


class Naive(BaseForecaster):
    name = "naive"

    def fit(self, y, *, period: int = 1):
        super().fit(y, period=period)
        self.fitted_[1:] = self.y[:-1]
        return self

    def forecast(self, h):
        return np.full(h, self.y[-1])

    def params(self):
        return {"_model": "Naive", "y": self.y.tolist(), "period": self.period}

    @classmethod
    def from_params(cls, p):
        return cls().fit(p["y"], period=p.get("period", 1))


class SeasonalNaive(BaseForecaster):
    name = "seasonal-naive"

    def fit(self, y, *, period: int = 1):
        super().fit(y, period=period)
        m = period
        self.fitted_[m:] = self.y[:-m]
        return self

    def forecast(self, h):
        m = self.period
        return np.array([self.y[-m + (i % m)] for i in range(h)])

    def params(self):
        return {"_model": "SeasonalNaive", "y": self.y.tolist(), "period": self.period}

    @classmethod
    def from_params(cls, p):
        return cls().fit(p["y"], period=p.get("period", 1))


class MovingAverage(BaseForecaster):
    name = "moving-average"

    def __init__(self, window: int = 3):
        self.window = window

    def fit(self, y, *, period: int = 1):
        super().fit(y, period=period)
        w = self.window
        for t in range(w, len(self.y)):
            self.fitted_[t] = self.y[t - w:t].mean()
        return self

    def forecast(self, h):
        return np.full(h, self.y[-self.window:].mean())

    def params(self):
        return {"_model": "MovingAverage", "y": self.y.tolist(), "period": self.period,
                "window": self.window}

    @classmethod
    def from_params(cls, p):
        return cls(window=p["window"]).fit(p["y"], period=p.get("period", 1))


class SimpleExponentialSmoothing(BaseForecaster):
    name = "ses"

    def __init__(self, alpha: float | None = None):
        self.alpha = alpha

    def fit(self, y, *, period: int = 1):
        super().fit(y, period=period)
        self.alpha_ = self.alpha if self.alpha is not None else _best_ses(self.y)
        s = self.y[0]
        for t in range(1, len(self.y)):
            self.fitted_[t] = s
            s = self.alpha_ * self.y[t] + (1 - self.alpha_) * s
        self.level_ = s
        return self

    def forecast(self, h):
        return np.full(h, self.level_)

    def params(self):
        return {"_model": "SimpleExponentialSmoothing", "y": self.y.tolist(),
                "period": self.period, "alpha": self.alpha_}

    @classmethod
    def from_params(cls, p):
        return cls(alpha=p["alpha"]).fit(p["y"], period=p.get("period", 1))


class Holt(BaseForecaster):
    name = "holt"

    def __init__(self, alpha: float | None = None, beta: float | None = None):
        self.alpha = alpha
        self.beta = beta

    def fit(self, y, *, period: int = 1):
        super().fit(y, period=period)
        if self.alpha is None or self.beta is None:
            a, b = _best_holt(self.y)
            self.alpha_ = self.alpha if self.alpha is not None else a
            self.beta_ = self.beta if self.beta is not None else b
        else:
            self.alpha_, self.beta_ = self.alpha, self.beta
        level, trend = self.y[0], self.y[1] - self.y[0]
        for t in range(1, len(self.y)):
            self.fitted_[t] = level + trend
            new_level = self.alpha_ * self.y[t] + (1 - self.alpha_) * (level + trend)
            new_trend = self.beta_ * (new_level - level) + (1 - self.beta_) * trend
            level, trend = new_level, new_trend
        self.level_, self.trend_ = level, trend
        return self

    def forecast(self, h):
        return np.array([self.level_ + (i + 1) * self.trend_ for i in range(h)])

    def params(self):
        return {"_model": "Holt", "y": self.y.tolist(), "period": self.period,
                "alpha": self.alpha_, "beta": self.beta_}

    @classmethod
    def from_params(cls, p):
        return cls(alpha=p["alpha"], beta=p["beta"]).fit(p["y"], period=p.get("period", 1))


class HoltWinters(BaseForecaster):
    name = "holt-winters"

    def __init__(self, alpha=None, beta=None, gamma=None):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def fit(self, y, *, period: int = 1):
        super().fit(y, period=period)
        m = self.period
        if m < 2:
            raise ValueError("holt-winters requires period >= 2")
        if len(self.y) < 2 * m:
            raise ValueError(f"need at least {2 * m} observations for period={m}")
        if self.alpha is None or self.beta is None or self.gamma is None:
            a, b, g = _best_hw(self.y, m)
            self.alpha_ = self.alpha if self.alpha is not None else a
            self.beta_ = self.beta if self.beta is not None else b
            self.gamma_ = self.gamma if self.gamma is not None else g
        else:
            self.alpha_, self.beta_, self.gamma_ = self.alpha, self.beta, self.gamma

        level = self.y[:m].mean()
        trend = (self.y[m:2 * m].mean() - self.y[:m].mean()) / m
        seasonal = self.y[:m] - level
        seasonal -= seasonal.mean()
        for t in range(m, len(self.y)):
            self.fitted_[t] = level + trend + seasonal[t % m]
            new_level = self.alpha_ * (self.y[t] - seasonal[t % m]) + (1 - self.alpha_) * (level + trend)
            new_trend = self.beta_ * (new_level - level) + (1 - self.beta_) * trend
            seasonal[t % m] = self.gamma_ * (self.y[t] - new_level) + (1 - self.gamma_) * seasonal[t % m]
            level, trend = new_level, new_trend
        self.level_, self.trend_, self.seasonal_ = level, trend, seasonal
        return self

    def forecast(self, h):
        m = self.period
        return np.array([
            self.level_ + (i + 1) * self.trend_ + self.seasonal_[(len(self.y) + i) % m]
            for i in range(h)
        ])

    def params(self):
        return {"_model": "HoltWinters", "y": self.y.tolist(), "period": self.period,
                "alpha": self.alpha_, "beta": self.beta_, "gamma": self.gamma_}

    @classmethod
    def from_params(cls, p):
        return cls(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"]).fit(
            p["y"], period=p.get("period", 2))


class AR(BaseForecaster):
    name = "ar"

    def __init__(self, order: int | None = None, max_order: int = 12):
        self.order = order
        self.max_order = max_order

    def _design(self, y, p):
        n = len(y)
        X = np.ones((n - p, p + 1))
        for j in range(1, p + 1):
            X[:, j] = y[p - j:n - j]
        return X, y[p:]

    def fit(self, y, *, period: int = 1):
        super().fit(y, period=period)
        n = len(self.y)
        p_max = min(self.max_order, n // 3, n - 2)
        if self.order is not None:
            p = min(self.order, p_max)
        else:
            best = None
            for p in range(1, p_max + 1):
                X, target = self._design(self.y, p)
                coef, *_ = np.linalg.lstsq(X, target, rcond=None)
                sse = float(((target - X @ coef) ** 2).sum())
                aic = (n - p) * np.log(sse / (n - p) + EPS) + 2 * (p + 2)
                if best is None or aic < best[0]:
                    best = (aic, p)
            p = best[1]
        self.order_ = p
        X, target = self._design(self.y, p)
        self.coef_, *_ = np.linalg.lstsq(X, target, rcond=None)
        self.fitted_[p:] = X @ self.coef_
        return self

    def forecast(self, h):
        hist = list(self.y)
        out = []
        for _ in range(h):
            row = np.ones(self.order_ + 1)
            for j in range(1, self.order_ + 1):
                row[j] = hist[-j]
            val = float(row @ self.coef_)
            out.append(val)
            hist.append(val)
        return np.asarray(out)

    def params(self):
        return {"_model": "AR", "y": self.y.tolist(), "period": self.period,
                "order": self.order_, "coef": self.coef_.tolist()}

    @classmethod
    def from_params(cls, p):
        model = cls(order=p["order"]).fit(p["y"], period=p.get("period", 1))
        model.coef_ = np.asarray(p["coef"], dtype=float)
        return model


MODELS = {
    "naive": Naive,
    "seasonal-naive": SeasonalNaive,
    "moving-average": MovingAverage,
    "ses": SimpleExponentialSmoothing,
    "holt": Holt,
    "holt-winters": HoltWinters,
    "ar": AR,
}


def make_model(name: str, *, period: int = 1, **kwargs) -> BaseForecaster:
    key = name.lower()
    if key not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    if key in ("seasonal-naive", "holt-winters") and period < 2:
        raise ValueError(f"{key} requires period >= 2")
    return MODELS[key](**kwargs)


# --- backtesting --------------------------------------------------------------

def backtest(y, model_factory, *, horizon: int = 6, period: int = 1,
             min_train: int | None = None, step: int = 1, max_origins: int = 40) -> dict:
    """Rolling-origin (expanding window) backtest. `model_factory` is a zero-arg
    callable returning a fresh, unfitted model."""
    y = _as_1d(y)
    n = len(y)
    start = min_train or max(period + 2, int(0.5 * n))
    origins = list(range(start, n - horizon + 1, step))
    if len(origins) > max_origins:
        origins = origins[:: max(1, len(origins) // max_origins)]
    if not origins:
        raise ValueError("series too short for the requested horizon/min_train")

    scale = naive_scale(y, period if period > 1 else 1)
    actuals, preds = [], []
    per_origin = []
    for o in origins:
        model = model_factory()
        model.fit(y[:o], period=period)
        fc = model.forecast(horizon)
        truth = y[o:o + horizon]
        actuals.extend(truth)
        preds.extend(fc)
        per_origin.append({
            "origin": int(o),
            "mae": round(float(np.mean(np.abs(truth - fc))), 4),
        })
    metrics = _metrics(actuals, preds, naive_scale=scale)
    metrics = {k: (round(v, 4) if v is not None else None) for k, v in metrics.items()}
    return {"horizon": horizon, "n_origins": len(origins), "metrics": metrics,
            "per_origin": per_origin}


def compare_models(y, *, horizon: int = 6, period: int = 1, names=None, **kwargs) -> dict:
    names = names or list(MODELS)
    results = {}
    for name in names:
        if name in ("seasonal-naive", "holt-winters") and period < 2:
            continue
        try:
            factory = lambda name=name: make_model(name, period=period)
            results[name] = backtest(y, factory, horizon=horizon, period=period, **kwargs)
        except ValueError as err:
            results[name] = {"error": str(err)}
    ranked = sorted(
        (n for n in results if "metrics" in results[n]),
        key=lambda n: results[n]["metrics"]["rmse"],
    )
    return {"results": results, "ranking": ranked}


# --- persistence --------------------------------------------------------------

_REGISTRY = {m.__name__: m for m in MODELS.values()}


def save_model(model: BaseForecaster, path) -> str:
    Path(path).write_text(json.dumps(model.params(), indent=2), encoding="utf-8")
    return str(path)


def load_model(path) -> BaseForecaster:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tag = data.get("_model")
    if tag not in _REGISTRY:
        raise ValueError(f"unknown model type: {tag!r}")
    return _REGISTRY[tag].from_params(data)