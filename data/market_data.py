"""
data/market_data.py -- the join between Alpaca's live data and the
pricing_engine's OptionQuote/MarketSmilePoint contracts.

Two Alpaca calls feed every contract:
  - GET /v2/options/contracts (Trading API): the master record -- strike,
    expiry, type, tradable, open_interest. Authoritative for "does this
    contract exist and can I trade it."
  - GET /v1beta1/options/snapshots/{symbol} (Data API): the live quote
    (bid/ask/last) and, when the feed provides it, Alpaca's own Greeks
    and implied vol. Authoritative for "what is it worth right now."

We join on contract symbol. Where Alpaca's own IV/Greeks are present we
keep them for reference, but every OptionQuote is *also* solvable via
pricing_engine.calibration.iv_solver against the mid price -- so the
signal layer never silently depends on one feed's Greek convention.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from alpaca_client import AlpacaClient
from pricing_engine.market_data.contract import MarketSmilePoint, OptionQuote, OptionType
from pricing_engine.calibration.iv_solver import solve_implied_vol, IVSolverError
from pricing_engine.models.bsm import BlackScholesMerton

_bsm = BlackScholesMerton()

TRADING_DAYS_PER_YEAR = 252


@dataclass
class ChainContract:
    symbol: str
    underlying: str
    strike: float
    expiry: dt.date
    option_type: OptionType
    open_interest: int
    tradable: bool
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    alpaca_iv: Optional[float]
    alpaca_delta: Optional[float]
    alpaca_gamma: Optional[float]
    alpaca_theta: Optional[float]
    alpaca_vega: Optional[float]

    @property
    def mid(self) -> Optional[float]:
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return 0.5 * (self.bid + self.ask)
        return self.last


def _years_to_expiry(expiry: dt.date, as_of: Optional[dt.date] = None) -> float:
    as_of = as_of or dt.date.today()
    return max((expiry - as_of).days, 0) / 365.0


def fetch_chain(
    client: AlpacaClient,
    underlying_symbol: str,
    expiration_date_gte: Optional[str] = None,
    expiration_date_lte: Optional[str] = None,
    option_type: Optional[str] = None,
    min_open_interest: int = 0,
) -> list[ChainContract]:
    """One call each to the contracts endpoint and the snapshot endpoint,
    joined into a flat list of ChainContract. Contracts with no live quote
    (illiquid, halted) are dropped -- nothing downstream should have to
    special-case a None mid price."""
    contracts_meta = client.get_option_contracts(
        underlying_symbol,
        expiration_date_gte=expiration_date_gte,
        expiration_date_lte=expiration_date_lte,
    )
    snapshots = client.get_option_chain_snapshot(
        underlying_symbol,
        option_type=option_type,
        expiration_date_gte=expiration_date_gte,
        expiration_date_lte=expiration_date_lte,
    )

    out: list[ChainContract] = []
    for c in contracts_meta:
        if not c.get("tradable", False):
            continue
        oi = int(c.get("open_interest") or 0)
        if oi < min_open_interest:
            continue
        snap = snapshots.get(c["symbol"])
        if not snap:
            continue
        quote = snap.get("latestQuote") or {}
        trade = snap.get("latestTrade") or {}
        greeks = snap.get("greeks") or {}
        bid, ask = quote.get("bp"), quote.get("ap")
        last = trade.get("p")
        if not ((bid and ask) or last):
            continue
        out.append(
            ChainContract(
                symbol=c["symbol"],
                underlying=underlying_symbol,
                strike=float(c["strike_price"]),
                expiry=dt.date.fromisoformat(c["expiration_date"]),
                option_type=OptionType.CALL if c["type"] == "call" else OptionType.PUT,
                open_interest=oi,
                tradable=True,
                bid=bid, ask=ask, last=last,
                alpaca_iv=snap.get("impliedVolatility"),
                alpaca_delta=greeks.get("delta"),
                alpaca_gamma=greeks.get("gamma"),
                alpaca_theta=greeks.get("theta"),
                alpaca_vega=greeks.get("vega"),
            )
        )
    return out


def to_option_quote(
    contract: ChainContract, spot: float, risk_free_rate: float, dividend_yield: float = 0.0,
) -> OptionQuote:
    return OptionQuote(
        underlying=contract.underlying,
        spot=spot,
        strike=contract.strike,
        expiry_years=_years_to_expiry(contract.expiry),
        option_type=contract.option_type,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        market_price=contract.mid,
        bid=contract.bid,
        ask=contract.ask,
        volume=None,
        open_interest=contract.open_interest,
        implied_vol=contract.alpaca_iv,
    )


def implied_vol_for(quote: OptionQuote, fallback_guess: float = 0.25) -> Optional[float]:
    """Solve for IV from the mid price if Alpaca didn't supply one (the
    indicative feed often doesn't). Returns None if unsolvable (crossed
    or stale quote) rather than raising -- callers should skip the strike."""
    if quote.implied_vol and quote.implied_vol > 0:
        return quote.implied_vol
    if not quote.market_price or quote.market_price <= 0 or quote.expiry_years <= 0:
        return None
    try:
        return solve_implied_vol(_bsm, quote, quote.market_price, initial_guess=fallback_guess).implied_vol
    except IVSolverError:
        return None


def build_smile(
    contracts: list[ChainContract], spot: float, risk_free_rate: float, dividend_yield: float = 0.0,
) -> dict[dt.date, list[MarketSmilePoint]]:
    """Group live IVs by expiry into MarketSmilePoint lists, ready for
    pricing_engine.calibration.local_vol_surface.SmileSlice.fit(...)."""
    by_expiry: dict[dt.date, list[MarketSmilePoint]] = {}
    for c in contracts:
        quote = to_option_quote(c, spot, risk_free_rate, dividend_yield)
        iv = implied_vol_for(quote)
        if iv is None:
            continue
        by_expiry.setdefault(c.expiry, []).append(
            MarketSmilePoint(strike=c.strike, expiry_years=quote.expiry_years, implied_vol=iv)
        )
    return by_expiry


def atm_implied_vol(contracts: list[ChainContract], spot: float, risk_free_rate: float) -> Optional[float]:
    """Straddle-relevant single number: average call+put IV at the strike
    closest to spot, across the nearest expiry present in `contracts`."""
    if not contracts:
        return None
    nearest_expiry = min(c.expiry for c in contracts)
    same_expiry = [c for c in contracts if c.expiry == nearest_expiry]
    atm_strike = min({c.strike for c in same_expiry}, key=lambda k: abs(k - spot))
    ivs = []
    for c in same_expiry:
        if c.strike != atm_strike:
            continue
        quote = to_option_quote(c, spot, risk_free_rate)
        iv = implied_vol_for(quote)
        if iv:
            ivs.append(iv)
    return float(np.mean(ivs)) if ivs else None


def daily_log_returns_from_bars(bars: list[dict]) -> np.ndarray:
    closes = np.array([b["c"] for b in bars], dtype=float)
    if len(closes) < 2:
        return np.array([])
    return np.diff(np.log(closes))
