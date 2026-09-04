"""
research/vol_forecast.py -- GARCH(1,1) + HAR-RV blended volatility
forecast, ported from IITfinal.ipynb (Part 1) and generalized off NVDA.

This produces the "what do we think realized vol will be" leg of the
volatility risk premium signal in research/signal.py. The other leg --
"what does the market currently think" -- comes from data/market_data.py
reading Alpaca's live option chain, replacing the notebook's
`realized_vol * 1.15` proxy (explicitly flagged as the #1 limitation in
OPTION-HEDGING-MODEL's own README) with the real thing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS_PER_YEAR = 252


@dataclass
class VolForecast:
    garch_conditional_vol: pd.Series   # annualized, in-sample conditional vol
    garch_next_step: float             # annualized, true one-step-ahead forecast
    har_forecast: pd.Series            # annualized HAR-RV blend
    blended: pd.Series                 # 50/50 GARCH + HAR, annualized
    blended_next: float                # today's blended point estimate


def har_rv(realized_vol: pd.Series) -> pd.Series:
    """Heterogeneous Autoregressive RV: 1-day / 1-week / 1-month realized
    vol blend (Corsi 2009), same weights as the original notebook."""
    rv_1d = realized_vol
    rv_1w = realized_vol.rolling(5).mean()
    rv_1m = realized_vol.rolling(22).mean()
    return 0.5 * rv_1d + 0.3 * rv_1w + 0.2 * rv_1m


def forecast_volatility(log_returns: pd.Series, realized_vol: pd.Series) -> VolForecast:
    """log_returns, realized_vol: aligned Series, same index as the price
    history used elsewhere (e.g. research.regime.detect_regime's input).
    realized_vol should already be annualized (21-day rolling std * sqrt(252),
    matching research/regime.py's convention) so HAR and GARCH outputs are
    directly comparable."""
    if len(log_returns) < 60:
        raise ValueError(f"need >=60 return observations for a stable GARCH fit, got {len(log_returns)}")

    returns_pct = log_returns.dropna() * 100  # arch expects returns in %, standard convention
    garch = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
    garch_fit = garch.fit(disp="off")

    garch_cond_vol = garch_fit.conditional_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) / 100
    garch_cond_vol.index = returns_pct.index

    one_step = garch_fit.forecast(horizon=1)
    garch_next = float(np.sqrt(one_step.variance.values[-1][0]) * np.sqrt(TRADING_DAYS_PER_YEAR) / 100)

    har = har_rv(realized_vol)
    idx = garch_cond_vol.index.intersection(har.index)
    blended = 0.5 * har.loc[idx] + 0.5 * garch_cond_vol.loc[idx]

    return VolForecast(
        garch_conditional_vol=garch_cond_vol,
        garch_next_step=garch_next,
        har_forecast=har,
        blended=blended,
        blended_next=float(blended.iloc[-1]) if len(blended) else garch_next,
    )


def estimate_hurst(log_realized_vol: "np.ndarray | pd.Series", lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32)) -> Optional[float]:
    """Gatheral-Jaisson-Rosenbaum (2018) log-regression Hurst estimator,
    ported from rough_vol_project_v2/code/fbm_simulator.py's
    estimate_hurst (same closed-form: regress log(mean squared lag-k
    increment) on log(k); H = slope / 2).

    DIAGNOSTIC ONLY -- deliberately NOT wired into forecast_volatility's
    blend. Before adding it there, a rough-vol-kernel forecast (weights
    ~ lag^(H-0.5) applied to this same log-RV path) was tested
    out-of-sample against the real SPY price history this project
    already ships (backtest/out/SPY_backtest.csv): it did not improve on
    the existing GARCH+HAR blend -- RMSE got WORSE at every blend
    weight tried, monotonically, with the empirically best weight on the
    rough-vol component landing at 0.0 (i.e. "don't use it"). Two
    reasons, both worth recording rather than silently dropping the
    idea: (1) applying this estimator to an already 21-day-ROLLING
    (i.e. overlapping-averaged) realized-vol series is a methodological
    mismatch with the literature it's from -- GJR's own results are on
    much higher-frequency vol proxies, and rolling-averaging manufactures
    artificial smoothness (measured H ~= 0.57 here on daily-close-only
    SPY data, vs. the ~0.1 the literature reports from intraday data);
    (2) even setting that aside, real roughness estimation is known to
    need higher-frequency (e.g. 5-minute) data than a single daily close
    per day provides, which this project's data source doesn't have.
    Exposed here purely as an informational read (dashboard: "vol path
    roughness") -- H well below ~0.5 would suggest genuine short-memory
    structure worth revisiting as a forecast input LATER, with proper
    higher-frequency data; H close to the ~0.5-0.6 typically seen on
    daily-close-derived series is exactly the "don't trust this as rough
    vol" signal that motivated keeping it out of position sizing."""
    x = np.asarray(log_realized_vol, dtype=float)
    x = x[~np.isnan(x)]
    log_vars, log_lags = [], []
    for lag in lags:
        if lag >= len(x):
            continue
        diffs = x[lag:] - x[:-lag]
        mean_sq = float(np.mean(diffs ** 2))
        if mean_sq <= 0:
            continue
        log_vars.append(np.log(mean_sq))
        log_lags.append(np.log(lag))
    if len(log_lags) < 3:
        return None
    slope = float(np.polyfit(log_lags, log_vars, 1)[0])
    return slope / 2.0


def vrp_zscore(forecast_vol: pd.Series, implied_vol: pd.Series, lookback: int = 60) -> pd.Series:
    """Volatility Risk Premium, z-scored: VRP = forecast - implied. A
    positive z-score means the market is under-pricing volatility relative
    to the model's forecast (options look cheap -- see research/signal.py
    for how this becomes a trade direction)."""
    vrp = forecast_vol - implied_vol
    return (vrp - vrp.rolling(lookback).mean()) / vrp.rolling(lookback).std()
