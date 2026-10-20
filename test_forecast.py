"""Tests for forecaster: decomposition, ACF/PACF, smoothing & AR models,
metrics, backtesting, persistence, CLI, and the Flask API."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from forecast import (
    AR,
    Holt,
    HoltWinters,
    MODELS,
    MovingAverage,
    Naive,
    SeasonalNaive,
    SimpleExponentialSmoothing,
    acf,
    backtest,
    centered_moving_average,
    compare_models,
    decompose,
    load_model,
    make_model,
    naive_scale,
    pacf,
    save_model,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def ar1(phi, n, seed=0, sigma=0.5):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal(scale=sigma)
    return y


def seasonal_series(n=120, period=12, seed=0):
    t = np.arange(n)
    return 20 + 0.3 * t + 5 * np.sin(2 * np.pi * t / period) + np.random.default_rng(seed).normal(scale=0.4, size=n)


class DecompositionTest(unittest.TestCase):
    def test_centered_ma_odd_window(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = centered_moving_average(y, 3)
        self.assertTrue(np.isnan(out[0]) and np.isnan(out[-1]))
        np.testing.assert_allclose(out[1:4], [2.0, 3.0, 4.0])

    def test_centered_ma_even_window(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = centered_moving_average(y, 4)
        # half-weighted at the ends: (0.5*y[t-2] + y[t-1] + y[t] + y[t+1] + 0.5*y[t+2]) / 4
        self.assertAlmostEqual(out[2], (0.5 * 1 + 2 + 3 + 4 + 0.5 * 5) / 4)
        self.assertAlmostEqual(out[3], (0.5 * 2 + 3 + 4 + 5 + 0.5 * 6) / 4)

    def test_seasonal_component_repeats(self):
        d = decompose(seasonal_series(), 12)
        np.testing.assert_allclose(d["seasonal"][:12], d["seasonal"][12:24], atol=1e-9)

    def test_seasonal_strength_high(self):
        d = decompose(seasonal_series(), 12)
        self.assertGreater(d["seasonal_strength"], 0.9)

    def test_seasonal_strength_low_for_noise(self):
        rng = np.random.default_rng(1)
        d = decompose(rng.normal(size=120), 12)
        self.assertLess(d["seasonal_strength"], 0.4)

    def test_period_too_small_raises(self):
        with self.assertRaises(ValueError):
            decompose(np.arange(50.0), 1)

    def test_series_too_short_raises(self):
        with self.assertRaises(ValueError):
            decompose(np.arange(10.0), 12)


class AcfPacfTest(unittest.TestCase):
    def test_acf_lag_zero_is_one(self):
        self.assertAlmostEqual(acf(seasonal_series(), 10)[0], 1.0)

    def test_acf_negative_on_alternating(self):
        alt = np.array([1.0, -1.0] * 40)
        self.assertLess(acf(alt, 1)[1], -0.9)

    def test_pacf_dominant_first_lag_for_ar1(self):
        p = pacf(ar1(0.8, 400, seed=2), 5)
        self.assertGreater(abs(p[1]), 0.6)
        self.assertLess(abs(p[2]), abs(p[1]))

    def test_naive_scale(self):
        y = np.array([1.0, 2.0, 4.0, 7.0])
        self.assertAlmostEqual(naive_scale(y, 1), np.mean([1, 2, 3]))


class ModelTest(unittest.TestCase):
    def test_naive_forecast_is_last(self):
        y = np.array([1.0, 2.0, 3.0, 9.0])
        fc = Naive().fit(y).forecast(3)
        np.testing.assert_allclose(fc, [9.0, 9.0, 9.0])

    def test_seasonal_naive_repeats(self):
        y = np.arange(1.0, 13.0)  # 1..12, period 12
        fc = SeasonalNaive().fit(y, period=4).forecast(4)
        np.testing.assert_allclose(fc, y[-4:])

    def test_moving_average_forecast(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fc = MovingAverage(window=3).fit(y).forecast(2)
        np.testing.assert_allclose(fc, [4.0, 4.0])  # mean of last 3 = 4

    def test_ses_constant_forecast(self):
        y = seasonal_series(60)
        fc = SimpleExponentialSmoothing(alpha=0.5).fit(y).forecast(5)
        self.assertTrue(np.allclose(fc, fc[0]))

    def test_ses_alpha_optimized_in_range(self):
        m = SimpleExponentialSmoothing().fit(seasonal_series(80))
        self.assertTrue(0.0 < m.alpha_ <= 1.0)

    def test_holt_continues_trend(self):
        y = np.arange(30, dtype=float) * 2
        fc = Holt(alpha=0.5, beta=0.5).fit(y).forecast(4)
        self.assertTrue(np.all(np.diff(fc) > 0))
        self.assertGreater(fc[-1], y[-1])

    def test_holt_winters_seasonality_in_forecast(self):
        m = HoltWinters(alpha=0.3, beta=0.1, gamma=0.5).fit(seasonal_series(), period=12)
        fc = m.forecast(24)
        self.assertGreater(fc.max() - fc.min(), 2.0)  # seasonal swing survives
        self.assertAlmostEqual(len(set(np.round(fc[:12], 6))), 12)

    def test_holt_winters_needs_period(self):
        with self.assertRaises(ValueError):
            make_model("holt-winters", period=1)

    def test_ar_recovers_coefficient(self):
        y = ar1(0.8, 500, seed=3)
        m = AR(order=1).fit(y)
        self.assertAlmostEqual(m.coef_[1], 0.8, delta=0.15)

    def test_ar_order_selected(self):
        m = AR(max_order=8).fit(ar1(0.7, 300, seed=4))
        self.assertGreaterEqual(m.order_, 1)

    def test_ar_forecast_shrinks_for_stable_ar(self):
        y = ar1(0.5, 400, seed=5)
        fc = AR(order=1).fit(y).forecast(5)
        self.assertEqual(len(fc), 5)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            make_model("banana")

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            Naive().fit([1.0])


class BacktestTest(unittest.TestCase):
    def test_backtest_naive_metrics(self):
        y = seasonal_series(80)
        r = backtest(y, lambda: Naive(), horizon=4, period=4, min_train=40, max_origins=10)
        self.assertGreater(r["n_origins"], 0)
        for key in ("mae", "rmse", "mape", "mase"):
            self.assertIn(key, r["metrics"])

    def test_backtest_too_short_raises(self):
        with self.assertRaises(ValueError):
            backtest(np.arange(5.0), lambda: Naive(), horizon=6, min_train=10)

    def test_compare_models_ranked(self):
        y = seasonal_series(80)
        res = compare_models(y, horizon=4, period=12, min_train=48, max_origins=6,
                             names=["naive", "ses", "moving-average"])
        self.assertTrue(res["ranking"])
        rmses = [res["results"][n]["metrics"]["rmse"] for n in res["ranking"]]
        self.assertEqual(rmses, sorted(rmses))

    def test_seasonal_model_better_than_naive_on_seasonal(self):
        y = seasonal_series(96, seed=1)
        naive = backtest(y, lambda: Naive(), horizon=12, period=12, min_train=60, max_origins=8)
        sn = backtest(y, lambda: SeasonalNaive(), horizon=12, period=12, min_train=60, max_origins=8)
        self.assertLess(sn["metrics"]["rmse"], naive["metrics"]["rmse"])


class PersistenceTest(unittest.TestCase):
    def _roundtrip(self, model, **fit_kwargs):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            save_model(model, path)
            return load_model(path)

    def test_naive_roundtrip(self):
        y = seasonal_series(40)
        loaded = self._roundtrip(Naive().fit(y))
        np.testing.assert_allclose(loaded.forecast(3), Naive().fit(y).forecast(3))

    def test_holt_winters_roundtrip(self):
        y = seasonal_series(60)
        m = HoltWinters(alpha=0.3, beta=0.1, gamma=0.5).fit(y, period=12)
        loaded = self._roundtrip(m)
        np.testing.assert_allclose(loaded.forecast(12), m.forecast(12))
        self.assertEqual(loaded.period, 12)

    def test_ar_roundtrip(self):
        y = ar1(0.6, 200, seed=6)
        m = AR(order=2).fit(y)
        loaded = self._roundtrip(m)
        np.testing.assert_allclose(loaded.coef_, m.coef_)
        np.testing.assert_allclose(loaded.forecast(3), m.forecast(3))

    def test_unknown_model_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"_model": "Nope"}, f)
            with self.assertRaises(ValueError):
                load_model(path)


class CliTest(unittest.TestCase):
    def _csv(self, d, n=96):
        path = os.path.join(d, "s.csv")
        y = seasonal_series(n, seed=2)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["month", "value"])
            for i, v in enumerate(y):
                w.writerow([i, round(float(v), 3)])
        return path

    def cli(self, *args):
        return subprocess.run([sys.executable, os.path.join(HERE, "cli.py"), *args],
                              capture_output=True, text=True, cwd=HERE)

    def test_analyze(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.cli("analyze", self._csv(d), "-c", "value", "-p", "12", "--lags", "6")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("ACF", r.stdout)
            self.assertIn("Seasonal strength", r.stdout)

    def test_forecast_writes_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "fc.csv")
            model = os.path.join(d, "m.json")
            r = self.cli("forecast", self._csv(d), "-c", "value", "-m", "holt-winters",
                         "-p", "12", "--horizon", "6", "--output", out, "--save", model)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as f:
                self.assertEqual(next(csv.reader(f)), ["step", "forecast", "lo80", "hi80", "lo95", "hi95"])
            self.assertEqual(load_model(model).name, "holt-winters")

    def test_forecast_from_saved_model(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m.json")
            self.cli("forecast", self._csv(d), "-c", "value", "-m", "ses",
                     "--horizon", "4", "--save", model)
            r = self.cli("forecast", "unused.csv", "--model-file", model, "--horizon", "4", "-q")
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_backtest_auto(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.cli("backtest", self._csv(d), "-c", "value", "-p", "12", "--horizon", "6")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("Best by RMSE", r.stdout)

    def test_demo(self):
        r = self.cli("demo", "-m", "holt-winters", "-p", "12", "--horizon", "6")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Forecast", r.stdout)

    def test_missing_column_returns_1(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.cli("forecast", self._csv(d), "-c", "nope")
            self.assertEqual(r.returncode, 1)
            self.assertIn("error:", r.stderr)

    def test_missing_file_returns_1(self):
        self.assertEqual(self.cli("analyze", "nope.csv").returncode, 1)


class FlaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as webapp
        webapp.app.config["TESTING"] = True
        cls.client = webapp.app.test_client()

    def _csv_text(self, n=96):
        y = seasonal_series(n, seed=9)
        return "month,value\n" + "\n".join(f"{i},{v:.3f}" for i, v in enumerate(y))

    def test_health(self):
        self.assertEqual(self.client.get("/health").get_json(), {"ok": True})

    def test_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"forecaster", r.data)

    def test_forecast_endpoint(self):
        r = self.client.post("/api/forecast", json={
            "csv": self._csv_text(), "column": "value", "model": "holt-winters",
            "period": 12, "horizon": 12})
        d = r.get_json()
        self.assertTrue(d["ok"], d)
        self.assertEqual(len(d["forecast"]), 12)
        self.assertEqual(len(d["history"]), 96)
        self.assertIn("decomposition", d)
        self.assertEqual(len(d["lo95"]), 12)

    def test_compare_flag(self):
        r = self.client.post("/api/forecast", json={
            "csv": self._csv_text(), "column": "value", "model": "ses",
            "period": 12, "horizon": 6, "compare": True})
        d = r.get_json()
        self.assertTrue(d["ok"], d)
        self.assertTrue(len(d.get("comparison", [])) >= 1)

    def test_bad_column_400(self):
        r = self.client.post("/api/forecast", json={
            "csv": self._csv_text(), "column": "nope", "model": "ses", "period": 12, "horizon": 4})
        self.assertEqual(r.status_code, 400)

    def test_empty_csv_400(self):
        r = self.client.post("/api/forecast", json={"csv": "", "column": "value", "model": "ses"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)