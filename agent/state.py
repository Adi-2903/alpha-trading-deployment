"""
agent/state.py -- the only persistence this project uses: a small JSON
file tracking the account equity high-water mark (for the drawdown kill
switch), a rolling log of recent decisions (for the API's /decisions
endpoint and the UI's activity feed), and -- new in v2 -- a rolling
equity curve, because "what is the agent's Sharpe ratio" was previously
unanswerable from live state at all: v1 persisted the high-water mark
(a single running max) but never a time series, so there was nothing to
compute a live Sharpe/Sortino/drawdown FROM. compute_risk_metrics below
is what api/server.py's new /metrics endpoint and the dashboard's risk
panel actually read.

This is intentionally not a database. Swap it for one before running
this unattended for weeks -- see README "Productionizing" section --
but for a hackathon-length paper-trading loop, a JSON file that's
trivial to inspect (`cat agent_state.json`) is the right amount of
infrastructure.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import os
STATE_PATH = Path("/tmp/agent_state.json") if os.environ.get("VERCEL") else Path(__file__).resolve().parent.parent / "agent_state.json"
MAX_DECISION_LOG = 200
MAX_EQUITY_CURVE = 5000
TRADING_PERIODS_PER_YEAR = 252


@dataclass
class AgentState:
    equity_high_water_mark: float = 0.0
    last_run_utc: str = ""
    decision_log: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)  # [{"t": iso_ts, "equity": float}, ...]


def load_state() -> AgentState:
    if not STATE_PATH.exists():
        return AgentState()
    try:
        raw = json.loads(STATE_PATH.read_text())
        raw.setdefault("equity_curve", [])
        return AgentState(**raw)
    except (json.JSONDecodeError, TypeError):
        return AgentState()


def save_state(state: AgentState) -> None:
    state.decision_log = state.decision_log[-MAX_DECISION_LOG:]
    state.equity_curve = state.equity_curve[-MAX_EQUITY_CURVE:]
    STATE_PATH.write_text(json.dumps(asdict(state), indent=2, default=str))


def record_decision(state: AgentState, entry: dict[str, Any]) -> None:
    state.decision_log.append(entry)


def record_equity(state: AgentState, timestamp_utc: str, equity: float) -> None:
    """Appends one point per agent cycle. Deliberately per-CYCLE, not
    per-calendar-day: at the default 15-minute loop interval this
    naturally gives an intraday-resolution curve early on, and
    compute_risk_metrics annualizes correctly either way because it
    infers the periods-per-year from the ACTUAL observed spacing of the
    points rather than assuming daily bars."""
    state.equity_curve.append({"t": timestamp_utc, "equity": float(equity)})


def _periods_per_year(timestamps: list[str]) -> float:
    """Infer an annualization factor from the equity curve's own
    observed spacing rather than assuming daily -- a curve logged every
    15 minutes during market hours and a curve logged once a day need
    very different sqrt(N) scalings, and hard-coding 252 (as
    backtest/run_backtest.py correctly does for its genuinely daily
    series) would silently misstate live Sharpe by orders of magnitude."""
    import datetime as dt
    if len(timestamps) < 3:
        return TRADING_PERIODS_PER_YEAR
    try:
        parsed = [dt.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in timestamps]
    except ValueError:
        return TRADING_PERIODS_PER_YEAR
    deltas = [(b - a).total_seconds() for a, b in zip(parsed, parsed[1:])]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        return TRADING_PERIODS_PER_YEAR
    median_seconds = sorted(deltas)[len(deltas) // 2]
    trading_day_seconds = 6.5 * 3600  # ~6.5h regular session
    if median_seconds <= trading_day_seconds:
        # multiple observations per trading day (the common case at the
        # default 15-minute loop interval): scale by how many such
        # gaps fit in a trading day, then by 252 trading days/year.
        return max(TRADING_PERIODS_PER_YEAR * trading_day_seconds / median_seconds, 1.0)
    # coarser than one trading session apart: treat the gap as N calendar
    # days and scale 252 trading-days/year by it directly (this is the
    # branch backtest/run_backtest.py's own hard-coded 252 corresponds
    # to, at N=1 -- genuinely once-a-day sampling).
    median_days = median_seconds / 86400.0
    return max(TRADING_PERIODS_PER_YEAR / median_days, 1.0)


def compute_risk_metrics(state: AgentState, min_points: int = 5) -> dict[str, Optional[float]]:
    """Sharpe/Sortino/max-drawdown/win-rate off the live equity_curve.
    Returns Nones (not zeros -- a young agent with 3 logged cycles has
    an UNKNOWN Sharpe, not a zero one) until there's enough history to
    say anything. Same rf=0 convention run_backtest.py already uses for
    the excess-return proxy (returns net of a flat 0% -- both this and
    the backtest treat period-over-period equity return itself as the
    excess return, not re-subtracting a separate risk-free drag that's
    already implicitly inside the account's own cash balance)."""
    points = state.equity_curve
    if len(points) < min_points:
        return {
            "sharpe": None, "sortino": None, "max_drawdown": None, "win_rate": None,
            "n_observations": len(points), "total_return": None,
        }
    equities = [p["equity"] for p in points]
    timestamps = [p["t"] for p in points]
    returns = [(b - a) / a for a, b in zip(equities, equities[1:]) if a > 0]
    if len(returns) < min_points - 1:
        return {
            "sharpe": None, "sortino": None, "max_drawdown": None, "win_rate": None,
            "n_observations": len(points), "total_return": None,
        }

    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / max(len(returns) - 1, 1)
    std_r = math.sqrt(var_r)
    periods_per_year = _periods_per_year(timestamps)
    sharpe = (mean_r / std_r) * math.sqrt(periods_per_year) if std_r > 1e-12 else None

    downside = [r for r in returns if r < 0]
    if len(downside) > 1:
        down_var = sum(r * r for r in downside) / len(downside)
        down_std = math.sqrt(down_var)
        sortino = (mean_r / down_std) * math.sqrt(periods_per_year) if down_std > 1e-12 else None
    else:
        sortino = None

    cum = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cum *= (1.0 + r)
        peak = max(peak, cum)
        max_dd = min(max_dd, cum / peak - 1.0)

    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    total_return = (equities[-1] - equities[0]) / equities[0] if equities[0] else None

    return {
        "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_dd, "win_rate": win_rate,
        "n_observations": len(points), "total_return": total_return,
    }
