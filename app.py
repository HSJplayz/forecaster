"""Web front-end for forecaster (Flask). Fits a smoothing or AR model on pasted
CSV and returns chart-ready JSON: history + fitted + forecast with intervals,
(classical) decomposition, ACF/PACF, and a model-comparison table. All charts
are rendered client-side as inline SVG; no CDN, no network."""

from __future__ import annotations

import csv
import io

import numpy as np
from flask import Flask, jsonify, render_template, request

from forecast import (
    MODELS,
    acf,
    compare_models,
    decompose,
    make_model,
    pacf,
)

app = Flask(__name__)


def _numeric_cols(rows):
    cols = []
    for name in rows[0].keys():
        try:
            [float(r[name]) for r in rows]
            cols.append(name)
        except (TypeError, ValueError):
            continue
    return cols


def _clean(a):
    a = np.asarray(a, dtype=float)
    return [None if np.isnan(v) else round(float(v), 4) for v in a]


def _residual_std(model):
    if getattr(model, "fitted_", None) is None:
        return 0.0
    mask = ~np.isnan(model.fitted_)
    if not mask.any():
        return 0.0
    return float(np.std(model.y[mask] - model.fitted_[mask]))


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/forecast")
def api_forecast():
    try:
        body = request.get_json(force=True, silent=True) or {}
        text = body.get("csv", "")
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            raise ValueError("CSV has no rows (or headers are missing)")
        cols = _numeric_cols(rows)
        column = body.get("column") or (cols[0] if cols else None)
        if not column or column not in rows[0]:
            raise ValueError(f"column {column!r} not found; numeric columns: {cols}")

        model_name = body.get("model", "holt-winters")
        if model_name not in MODELS:
            raise ValueError(f"unknown model {model_name!r}")
        period = int(body.get("period", 1))
        horizon = int(body.get("horizon", 12))
        do_compare = bool(body.get("compare", False))

        y = np.asarray([float(r[column]) for r in rows], dtype=float)
        if len(y) < 4:
            raise ValueError("need at least 4 observations")

        model = make_model(model_name, period=period)
        model.fit(y, period=period)
        fc = model.forecast(horizon)
        sigma = _residual_std(model)
        z = 1.96
        lo = [round(float(v - z * sigma), 4) for v in fc]
        hi = [round(float(v + z * sigma), 4) for v in fc]

        result = {
            "ok": True,
            "overview": {"rows": len(y), "column": column, "model": model.name,
                         "period": period, "horizon": horizon,
                         "residual_std": round(sigma, 4)},
            "n_train": len(y),
            "history": _clean(y),
            "fitted": _clean(model.fitted_),
            "forecast": _clean(fc),
            "lo95": lo,
            "hi95": hi,
            "acf": _clean(acf(y, min(24, len(y) - 1))),
            "pacf": _clean(pacf(y, min(24, len(y) - 1))),
        }
        if hasattr(model, "alpha_"):
            params = {"alpha": model.alpha_}
            if hasattr(model, "beta_"):
                params["beta"] = model.beta_
            if hasattr(model, "gamma_"):
                params["gamma"] = model.gamma_
            result["smoothing"] = {k: round(float(v), 3) for k, v in params.items()}
        if getattr(model, "order_", None):
            result["ar_order"] = model.order_

        if period >= 2 and len(y) >= 2 * period:
            d = decompose(y, period)
            result["decomposition"] = {
                "trend": _clean(d["trend"]),
                "seasonal": _clean(d["seasonal"]),
                "resid": _clean(d["resid"]),
                "seasonal_strength": round(d["seasonal_strength"], 4),
                "trend_strength": round(d["trend_strength"], 4),
            }

        if do_compare and len(y) >= 12:
            comp = compare_models(y, horizon=min(horizon, len(y) // 3), period=period)
            result["comparison"] = [
                {"model": n, **comp["results"][n]["metrics"]}
                for n in comp["ranking"]
            ]
        return jsonify(result)
    except Exception as err:
        return jsonify({"ok": False, "error": str(err)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)