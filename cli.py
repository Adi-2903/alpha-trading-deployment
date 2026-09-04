"""
cli.py -- single entry point for everything in this project.

    python cli.py run --once              # one agent cycle, respects DRY_RUN in .env
    python cli.py run --once --live       # force live execution for this run only
    python cli.py run --loop --interval 900   # repeat every 15 minutes
    python cli.py backtest --symbol NVDA --years 3
    python cli.py serve --port 8000       # launch the FastAPI backend for a UI
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("alpaca_vol_agent.cli")


def _settings_for_run(args: argparse.Namespace):
    from config import SETTINGS, Settings

    if not args.live:
        return SETTINGS
    log.warning("Running LIVE (dry_run=False) for this invocation -- real paper orders will submit.")
    return Settings(
        api_key=SETTINGS.api_key, api_secret=SETTINGS.api_secret,
        data_feed=SETTINGS.data_feed, stock_feed=SETTINGS.stock_feed, symbol=SETTINGS.symbol,
        target_dte_days=SETTINGS.target_dte_days, dte_tolerance_days=SETTINGS.dte_tolerance_days,
        short_leg_target_delta=SETTINGS.short_leg_target_delta, wing_target_delta=SETTINGS.wing_target_delta,
        dry_run=False, i_understand_this_is_live=SETTINGS.i_understand_this_is_live,
        enable_credit_spreads=SETTINGS.enable_credit_spreads,
        enable_equity_hedge=SETTINGS.enable_equity_hedge, min_open_interest=SETTINGS.min_open_interest,
        risk=SETTINGS.risk, hedge=SETTINGS.hedge,
    )


def cmd_run(args: argparse.Namespace) -> None:
    from agent.loop import run_cycle

    settings = _settings_for_run(args)

    def one_cycle():
        result = run_cycle(settings)
        print(json.dumps(dataclasses.asdict(result), indent=2, default=str))
        for note in result.notes:
            log.warning(note)

    if args.loop:
        log.info("Starting agent loop, interval=%ss, dry_run=%s, symbol=%s", args.interval, settings.dry_run, settings.symbol)
        while True:
            try:
                one_cycle()
            except Exception:
                log.exception("agent cycle failed; will retry next interval")
            time.sleep(args.interval)
    else:
        one_cycle()


def cmd_backtest(args: argparse.Namespace) -> None:
    from backtest.run_backtest import run_backtest

    summary = run_backtest(args.symbol, years=args.years, initial_capital=args.capital)
    print(json.dumps(dataclasses.asdict(summary), indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca volatility trading agent -- CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the agent")
    p_run.add_argument("--once", action="store_true", default=True, help="(default) run a single cycle")
    p_run.add_argument("--loop", action="store_true", help="run continuously")
    p_run.add_argument("--interval", type=int, default=900, help="seconds between cycles in --loop mode")
    p_run.add_argument("--live", action="store_true", help="override DRY_RUN=false for this invocation")
    p_run.set_defaults(func=cmd_run)

    p_bt = sub.add_parser("backtest", help="run the offline historical backtest")
    p_bt.add_argument("--symbol", default="SPY")
    p_bt.add_argument("--years", type=float, default=3.0)
    p_bt.add_argument("--capital", type=float, default=1_000_000.0)
    p_bt.set_defaults(func=cmd_backtest)

    p_serve = sub.add_parser("serve", help="launch the FastAPI backend")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
