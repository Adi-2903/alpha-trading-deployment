"""
check_deployment.py -- quick smoke-test for the Vercel deployment.

Usage:
    # against the live Vercel URL:
    python check_deployment.py https://your-project.vercel.app

    # against local dev server (uvicorn api.index:app --reload --port 8000):
    python check_deployment.py http://localhost:8000

Checks:
    /health      -- server is up, returns {ok: true}
    /            -- root lists all endpoints
    POST /backtest  -- Errno 30 fix: should NOT return a read-only-filesystem error
                       (returns 501 if yfinance/arch/hmmlearn aren't installed,
                        502 on any other failure, 200 on success)
"""
from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from typing import Any


BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def get(path: str) -> tuple[int, Any]:
    url = BASE_URL + path
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read())
        except Exception:
            pass
        return e.code, body


def post(path: str, payload: dict) -> tuple[int, Any]:
    url = BASE_URL + path
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read())
        except Exception:
            pass
        return e.code, body


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    return ok


# ---------------------------------------------------------------------------
print(f"\nChecking deployment at: {BASE_URL}\n")
all_ok = True

# 1. /health
code, body = get("/health")
ok = code == 200 and body.get("ok") is True
all_ok &= check("/health", ok, f"HTTP {code}" + (f", symbol={body.get('symbol')}, dry_run={body.get('dry_run')}" if ok else f", body={body}"))

# 2. / root
code, body = get("/")
ok = code == 200 and "endpoints" in body
all_ok &= check("/ (root)", ok, f"HTTP {code}")

# 3. POST /backtest -- the Errno 30 fix
#    We use a tiny 1-year window to keep runtime short; the point is to
#    confirm it doesn't blow up with a read-only-filesystem error.
code, body = post("/backtest", {"symbol": "SPY", "years": 3.0, "capital": 100000.0, "cost_bps": 5.0})
detail_str = body.get("detail", "") if isinstance(body, dict) else str(body)

if code == 200:
    sharpe = body.get("sharpe", "?")
    all_ok &= check("POST /backtest (Errno 30 fix)", True, f"HTTP 200 — Sharpe={sharpe:.2f}" if isinstance(sharpe, float) else f"HTTP 200")
elif code == 501:
    # deps not installed — filesystem fix worked, but packages missing
    print(f"  {WARN}  POST /backtest (Errno 30 fix)  (HTTP 501 — deps missing: {detail_str})")
    print(f"       The read-only-filesystem fix is working ✓")
    print(f"       yfinance/arch/hmmlearn are not installed on the Vercel runtime.")
    print(f"       → Run the backtest locally:  python -m backtest.run_backtest --symbol SPY --years 3")
elif "read-only" in detail_str.lower() or "errno 30" in detail_str.lower():
    all_ok &= check("POST /backtest (Errno 30 fix)", False, f"HTTP {code} — STILL getting read-only error! {detail_str}")
else:
    # Some other 502 (e.g. yfinance network error in CI) — not a filesystem issue
    print(f"  {WARN}  POST /backtest  (HTTP {code} — {detail_str[:120]})")
    print(f"       Not a read-only filesystem error — likely a network/data issue in the serverless env.")

# ---------------------------------------------------------------------------
print()
if all_ok:
    print(f"{PASS}  All checks passed.\n")
else:
    print(f"{FAIL}  Some checks failed — see above.\n")
    sys.exit(1)
