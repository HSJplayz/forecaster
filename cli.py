"""CLI for forecaster: analyze, forecast, backtest, demo."""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from forecast import (
    MODELS,
    acf,
    backtest,
    compare_models,
    decompose,
    load_model,
    make_model,
    naive_scale,
    pacf,
    save_model,
)


def _read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"'{path}' has no data rows")
    return rows


def _numeric_cols(rows):
    cols = []
    for name in rows[0].keys():
        try:
            [float(r[name]) for r in rows]
            cols.append(name)
        except (TypeError, ValueError):
            continue
    return cols


def load_series(path, column=None):
    rows = _read_rows(path)
    cols = _numeric_cols(rows)
    if column:
        if column not in rows[0]:
            raise ValueError(f"column {column!r} not found; numeric columns: {cols}")
        col = column
    else:
        if len(cols) == 1:
            col = cols[0]
        elif not cols:
            raise ValueError("no numeric columns found")
        else:
            raise ValueError(f"multiple numeric columns {cols}; pick one with -c")
    y = np.asarray([float(r[col]) for r in rows], dtype=float)
    return y, col


def residuals_std(model):
    if getattr(model, "fitted_", None) is None:
        return 0.0
    mask = ~np.isnan(model.fitted_)
    if not mask.any():
        return 0.0
    return float(np.std(model.y[mask] - model.fitted_[mask]))


def intervals(fc, sigma, z80=1.282, z95=1.96):
    return [(float(v - z80 * sigma), float(v + z80 * sigma),
             float(v - z95 * sigma), float(v + z95 * sigma)) for v in fc]


def build_parser():
    p = argparse.ArgumentParser(
        prog="forecaster",
        description="Time-series decomposition, exponential smoothing, and AR forecasting (numpy only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python cli.py analyze sales.csv -c value -p 12\n"
            "  python cli.py forecast sales.csv -c value -m holt-winters -p 12 --horizon 12\n"
            "  python cli.py forecast sales.csv -m ar --horizon 6 --output forecast.csv --save model.json\n"
            "  python cli.py backtest sales.csv -c value -p 12 --horizon 6\n"
            "  python cli.py demo -m holt-winters -p 12 --horizon 12"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="decomposition + ACF/PACF diagnostics")
    a.add_argument("input")
    a.add_argument("-c", "--column")
    a.add_argument("-p", "--period", type=int, default=1)
    a.add_argument("--lags", type=int, default=12)

    f = sub.add_parser("forecast", help="fit a model and forecast")
    f.add_argument("input")
    f.add_argument("-c", "--column")
    f.add_argument("-m", "--model", default="holt-winters", choices=sorted(MODELS))
    f.add_argument("-p", "--period", type=int, default=1)
    f.add_argument("--horizon", type=int, default=12)
    f.add_argument("--output", help="write forecasts to this CSV")
    f.add_argument("--save", help="save the fitted model to this JSON file")
    f.add_argument("--model-file", help="load a fitted model instead of training")
    f.add_argument("-q", "--quiet", action="store_true")

    b = sub.add_parser("backtest", help="rolling-origin backtest / model comparison")
    b.add_argument("input")
    b.add_argument("-c", "--column")
    b.add_argument("-m", "--model", default="auto")
    b.add_argument("-p", "--period", type=int, default=1)
    b.add_argument("--horizon", type=int, default=6)

    d = sub.add_parser("demo", help="forecast a synthetic trend + seasonal series")
    d.add_argument("-m", "--model", default="holt-winters", choices=sorted(MODELS))
    d.add_argument("-p", "--period", type=int, default=12)
    d.add_argument("-n", "--length", type=int, default=144)
    d.add_argument("--horizon", type=int, default=12)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("-q", "--quiet", action="store_true")
    return p


def cmd_analyze(args):
    y, col = load_series(args.input, args.column)
    print(f"Series '{col}': {len(y)} observations")
    print(f"Mean: {y.mean():.3f}   Std: {y.std():.3f}   Min/Max: {y.min():.3f}/{y.max():.3f}")
    if args.period >= 2:
        d = decompose(y, args.period)
        print(f"Period: {args.period}")
        print(f"Seasonal strength: {d['seasonal_strength']:.3f}   "
              f"Trend strength: {d['trend_strength']:.3f}")
        print(f"Residual std: {d['resid_std']:.3f}")
        order = np.argsort(-np.abs(d["seasonal"][:args.period]))
        print(f"Strongest seasonal phases: {[int(i) for i in order[:3]]}")
    a = acf(y, args.lags)
    pac = pacf(y, args.lags)
    print("ACF :", " ".join(f"{v:+.2f}" for v in a))
    print("PACF:", " ".join(f"{v:+.2f}" for v in pac))
    hint = "likely non-stationary - consider differencing" if a[1] > 0.7 else "no strong non-stationarity signal"
    print(f"Hint: {hint}")
    return 0


def cmd_forecast(args):
    if args.model_file:
        model = load_model(args.model_file)
        n = len(model.y)
        fc = model.forecast(args.horizon)
        sigma = residuals_std(model)
        print(f"Loaded {model.name} model ({n} observations), horizon {args.horizon}")
    else:
        y, col = load_series(args.input, args.column)
        model = make_model(args.model, period=args.period)
        model.fit(y, period=args.period)
        fc = model.forecast(args.horizon)
        sigma = residuals_std(model)

    if not args.quiet:
        print(f"Model: {model.name}"
              + (f"  period={model.period}" if model.period >= 2 else ""))
        if hasattr(model, "alpha_"):
            extra = f"alpha={model.alpha_:.2f}"
            for name in ("beta_", "gamma_"):
                if hasattr(model, name):
                    extra += f" {name[:-1]}={getattr(model, name):.2f}"
            print("Smoothing:", extra)
        if getattr(model, "order_", None):
            print(f"AR order: {model.order_}")
        print(f"Forecast (next {args.horizon}):")
        for i, v in enumerate(fc, 1):
            lo80, hi80, lo95, hi95 = intervals([v], sigma)[0]
            print(f"  +{i:>2}: {v:10.3f}  95%[{lo95:10.3f}, {hi95:10.3f}]")

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "forecast", "lo80", "hi80", "lo95", "hi95"])
            for i, v in enumerate(fc, 1):
                lo80, hi80, lo95, hi95 = intervals([v], sigma)[0]
                w.writerow([i, round(float(v), 6), round(lo80, 6), round(hi80, 6),
                            round(lo95, 6), round(hi95, 6)])
        if not args.quiet:
            print(f"Wrote {len(fc)} forecasts -> {args.output}")
    if args.save:
        save_model(model, args.save)
        if not args.quiet:
            print(f"Saved model -> {args.save}")
    return 0


def cmd_backtest(args):
    y, col = load_series(args.input, args.column)
    if args.model == "auto":
        res = compare_models(y, horizon=args.horizon, period=args.period)
        print(f"Backtest '{col}': {len(y)} obs, horizon {args.horizon}, period {args.period}")
        print(f"{'model':16s}{'MAE':>10}{'RMSE':>10}{'MAPE%':>9}{'MASE':>9}")
        for name in res["ranking"]:
            m = res["results"][name]["metrics"]
            print(f"{name:16s}{m['mae']:>10.3f}{m['rmse']:>10.3f}"
                  f"{(m['mape'] if m['mape'] is not None else float('nan')):>9.2f}"
                  f"{(m['mase'] if m['mase'] is not None else float('nan')):>9.3f}")
        print(f"Best by RMSE: {res['ranking'][0]}")
    else:
        factory = lambda: make_model(args.model, period=args.period)
        r = backtest(y, factory, horizon=args.horizon, period=args.period)
        m = r["metrics"]
        print(f"Model: {args.model}  origins: {r['n_origins']}  horizon: {args.horizon}")
        print(f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
              f"MAPE={m['mape']:.2f}%  MASE={m['mase']:.3f}")
    return 0


def cmd_demo(args):
    rng = np.random.default_rng(args.seed)
    t = np.arange(args.length)
    y = 20 + 0.3 * t + 5 * np.sin(2 * np.pi * t / args.period) + rng.normal(scale=0.5, size=args.length)
    model = make_model(args.model, period=args.period)
    model.fit(y, period=args.period)
    fc = model.forecast(args.horizon)
    if not args.quiet:
        print(f"Demo: {args.length} obs, trend + period-{args.period} seasonality, horizon {args.horizon}")
        print(f"Model: {model.name}")
        print("Forecast:", " ".join(f"{v:.2f}" for v in fc))
        print(f"Residual std: {residuals_std(model):.3f}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return {"analyze": cmd_analyze, "forecast": cmd_forecast,
                "backtest": cmd_backtest, "demo": cmd_demo}[args.command](args)
    except (ValueError, OSError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except Exception as err:  # pragma: no cover
        print(f"error: unexpected failure: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
