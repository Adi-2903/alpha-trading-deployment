"""
strategy/legs.py -- turns a sizing decision into concrete option
contracts, respecting what Alpaca's paper options levels actually allow.

This is the module that encodes a constraint worth stating plainly: per
docs.alpaca.markets/us/docs/options-trading, Alpaca's options levels
support Level 1 (covered call / cash-secured put), Level 2 (buy calls,
buy puts), and Level 3 (buy call spreads / buy put spreads via
order_class="mleg", where "all legs must be covered within the same
order" -- naked short legs are rejected even inside a multi-leg order).
There is no level that allows a naked short straddle or strangle.

So the two directions from strategy/sizing.py map to two different,
both fully compliant, structures:

  long_vol  (kelly > 0, buy volatility): a plain long straddle -- one
      single-leg BUY order for the ATM call, one for the ATM put. Level 2.
  short_vol (kelly < 0, sell volatility): an iron condor -- short strikes
      near `short_leg_target_delta`, long wings near `wing_target_delta`,
      submitted as ONE order_class="mleg" order with all four legs so
      Alpaca's "all legs covered in the same order" rule is satisfied by
      construction. Level 3. Every short leg in the structure is paired
      with a further-OTM long leg in the SAME order -- there is no
      uncovered leg at any point, including mid-fill (MLeg orders fill
      as a single unit or not at all).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

from data.market_data import ChainContract, implied_vol_for, to_option_quote
from pricing_engine.market_data.contract import OptionType
from pricing_engine.models.bsm import BlackScholesMerton

_bsm = BlackScholesMerton()


@dataclass
class StraddleLegs:
    expiry: dt.date
    call: ChainContract
    put: ChainContract
    price: float     # call mid + put mid, per share
    gamma: float      # per straddle (1 call + 1 put), per-underlying-unit convention
    vega: float


@dataclass
class IronCondorLegs:
    expiry: dt.date
    short_put: ChainContract
    long_put: ChainContract      # further OTM, protects the short put
    short_call: ChainContract
    long_call: ChainContract     # further OTM, protects the short call
    put_width: float
    call_width: float
    net_credit: float            # per share, positive = credit
    max_loss_per_contract: float  # dollars, conservative (ignores credit)
    contracts: int


def select_expiry(contracts: list[ChainContract], target_dte_days: int, tolerance_days: int, as_of: Optional[dt.date] = None) -> Optional[dt.date]:
    as_of = as_of or dt.date.today()
    expiries = sorted({c.expiry for c in contracts})
    if not expiries:
        return None
    best = min(expiries, key=lambda e: abs((e - as_of).days - target_dte_days))
    if abs((best - as_of).days - target_dte_days) > tolerance_days:
        return None
    return best


def _delta_of(contract: ChainContract, spot: float, risk_free_rate: float) -> Optional[float]:
    if contract.alpaca_delta is not None:
        return contract.alpaca_delta
    quote = to_option_quote(contract, spot, risk_free_rate)
    iv = implied_vol_for(quote)
    if iv is None or quote.expiry_years <= 0:
        return None
    return _bsm.delta(quote, iv)


def _nearest_by_abs_delta(
    candidates: list[ChainContract], target_abs_delta: float, spot: float, risk_free_rate: float,
) -> Optional[ChainContract]:
    scored = []
    for c in candidates:
        d = _delta_of(c, spot, risk_free_rate)
        if d is None:
            continue
        scored.append((abs(abs(d) - target_abs_delta), c))
    if not scored:
        return None
    return min(scored, key=lambda t: t[0])[1]


def build_straddle(
    chain: list[ChainContract], expiry: dt.date, spot: float, risk_free_rate: float,
) -> Optional[StraddleLegs]:
    same_expiry = [c for c in chain if c.expiry == expiry]
    calls = [c for c in same_expiry if c.option_type == OptionType.CALL]
    puts = [c for c in same_expiry if c.option_type == OptionType.PUT]
    if not calls or not puts:
        return None
    atm_strike = min({c.strike for c in calls} & {c.strike for c in puts}, key=lambda k: abs(k - spot), default=None)
    if atm_strike is None:
        return None
    call = next(c for c in calls if c.strike == atm_strike)
    put = next(c for c in puts if c.strike == atm_strike)
    if call.mid is None or put.mid is None:
        return None

    call_quote = to_option_quote(call, spot, risk_free_rate)
    put_quote = to_option_quote(put, spot, risk_free_rate)
    call_iv = implied_vol_for(call_quote)
    put_iv = implied_vol_for(put_quote)
    gamma = vega = 0.0
    if call_iv:
        gamma += _bsm.gamma(call_quote, call_iv)
        vega += _bsm.vega(call_quote, call_iv)
    if put_iv:
        gamma += _bsm.gamma(put_quote, put_iv)
        vega += _bsm.vega(put_quote, put_iv)

    return StraddleLegs(expiry=expiry, call=call, put=put, price=call.mid + put.mid, gamma=gamma, vega=vega)


def build_iron_condor(
    chain: list[ChainContract], expiry: dt.date, spot: float, risk_free_rate: float,
    short_delta: float, wing_delta: float, max_loss_budget: float, max_contracts: int,
) -> Optional[IronCondorLegs]:
    same_expiry = [c for c in chain if c.expiry == expiry]
    calls = sorted((c for c in same_expiry if c.option_type == OptionType.CALL), key=lambda c: c.strike)
    puts = sorted((c for c in same_expiry if c.option_type == OptionType.PUT), key=lambda c: c.strike)
    if len(calls) < 2 or len(puts) < 2:
        return None

    otm_calls = [c for c in calls if c.strike >= spot]
    otm_puts = [c for c in puts if c.strike <= spot]
    short_call = _nearest_by_abs_delta(otm_calls, short_delta, spot, risk_free_rate)
    short_put = _nearest_by_abs_delta(otm_puts, short_delta, spot, risk_free_rate)
    if short_call is None or short_put is None:
        return None

    further_calls = [c for c in otm_calls if c.strike > short_call.strike]
    further_puts = [c for c in otm_puts if c.strike < short_put.strike]
    long_call = _nearest_by_abs_delta(further_calls, wing_delta, spot, risk_free_rate) or (
        further_calls[-1] if further_calls else None
    )
    long_put = _nearest_by_abs_delta(further_puts, wing_delta, spot, risk_free_rate) or (
        further_puts[-1] if further_puts else None
    )
    if long_call is None or long_put is None:
        return None
    if any(c.mid is None for c in (short_call, long_call, short_put, long_put)):
        return None

    call_width = long_call.strike - short_call.strike
    put_width = short_put.strike - long_put.strike
    net_credit = (short_call.mid - long_call.mid) + (short_put.mid - long_put.mid)
    if net_credit <= 0:
        # A short iron condor should always collect a net credit (the
        # short inner strikes are worth more than the protective outer
        # wings). A non-positive net credit here means the quotes are
        # crossed/stale or the strike selection degenerated -- refuse to
        # build the order rather than submit a structure that pays a
        # debit to sell volatility, which would defeat the whole point.
        return None
    worst_width = max(call_width, put_width)
    max_loss_per_contract = max(worst_width * 100 - net_credit * 100, 0.0)
    if max_loss_per_contract <= 0:
        return None

    contracts = int(max_loss_budget // max_loss_per_contract)
    contracts = max(0, min(contracts, max_contracts))
    if contracts == 0:
        return None

    return IronCondorLegs(
        expiry=expiry, short_put=short_put, long_put=long_put, short_call=short_call, long_call=long_call,
        put_width=put_width, call_width=call_width, net_credit=net_credit,
        max_loss_per_contract=max_loss_per_contract, contracts=contracts,
    )
