"""
Tests for alpaca_client.py's CLI transport. These mock subprocess.run
entirely -- no real `alpaca` binary or network call happens here (this
sandbox can't reach alpaca.markets or install the Go binary anyway).
What's verified is the contract that matters for the hackathon's hard
requirement: every call shells out to `alpaca api ...` with the right
method/path/body/env, not to `requests` or any other bare HTTP client.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from config import Settings


def _settings(**overrides) -> Settings:
    base = dict(api_key="PKtest", api_secret="secrettest")
    base.update(overrides)
    return Settings(**base)


def _fake_run(stdout: str = "{}", stderr: str = "", returncode: int = 0):
    def _runner(args, input=None, capture_output=None, text=None, timeout=None, env=None):
        _runner.calls.append({"args": args, "input": input, "env": env})
        result = MagicMock()
        result.stdout, result.stderr, result.returncode = stdout, stderr, returncode
        return result
    _runner.calls = []
    return _runner


@patch("alpaca_client.shutil.which", return_value="/usr/local/bin/alpaca")
def test_client_requires_cli_binary_on_path(mock_which):
    from alpaca_client import AlpacaClient
    client = AlpacaClient(settings=_settings())
    assert client.cli_binary == "alpaca"


@patch("alpaca_client.shutil.which", return_value=None)
def test_client_raises_clear_error_when_cli_missing(mock_which):
    from alpaca_client import AlpacaClient
    with pytest.raises(RuntimeError, match="was not found on PATH"):
        AlpacaClient(settings=_settings())


@patch("alpaca_client.shutil.which", return_value="/usr/local/bin/alpaca")
def test_get_account_shells_out_to_alpaca_api(mock_which):
    from alpaca_client import AlpacaClient
    runner = _fake_run(stdout=json.dumps({"equity": "100000", "cash": "50000"}))
    with patch("alpaca_client.subprocess.run", side_effect=runner):
        client = AlpacaClient(settings=_settings())
        account = client.get_account()

    assert account["equity"] == "100000"
    call = runner.calls[0]
    assert call["args"][0] == "alpaca"
    assert call["args"][1:5] == ["api", "GET", "/v2/account", "--quiet"]
    assert call["env"]["ALPACA_API_KEY"] == "PKtest"
    assert call["env"]["ALPACA_SECRET_KEY"] == "secrettest"
    # never a live-trading override unless the project explicitly opted in
    assert "ALPACA_LIVE_TRADE" not in call["env"]


@patch("alpaca_client.shutil.which", return_value="/usr/local/bin/alpaca")
def test_submit_order_pipes_json_body_via_stdin(mock_which):
    from alpaca_client import AlpacaClient
    runner = _fake_run(stdout=json.dumps({"id": "abc123", "status": "accepted"}))
    with patch("alpaca_client.subprocess.run", side_effect=runner):
        client = AlpacaClient(settings=_settings())
        payload = {"symbol": "AAPL", "qty": "1", "side": "buy", "type": "market", "time_in_force": "day"}
        result = client.submit_order(payload)

    assert result["status"] == "accepted"
    call = runner.calls[0]
    assert call["args"][1:5] == ["api", "POST", "/v2/orders", "--quiet"]
    assert json.loads(call["input"]) == payload


@patch("alpaca_client.shutil.which", return_value="/usr/local/bin/alpaca")
def test_get_orders_builds_query_string_from_params(mock_which):
    from alpaca_client import AlpacaClient
    runner = _fake_run(stdout="[]")
    with patch("alpaca_client.subprocess.run", side_effect=runner):
        client = AlpacaClient(settings=_settings())
        client.get_orders(status="open", limit=50)

    path_arg = runner.calls[0]["args"][3]
    assert path_arg.startswith("/v2/orders?")
    assert "status=open" in path_arg
    assert "limit=50" in path_arg


@patch("alpaca_client.shutil.which", return_value="/usr/local/bin/alpaca")
def test_nonzero_exit_raises_with_stderr_body(mock_which):
    from alpaca_client import AlpacaClient, AlpacaCLIError
    runner = _fake_run(stdout="", stderr='{"error":"unauthorized","status":401}', returncode=2)
    with patch("alpaca_client.subprocess.run", side_effect=runner):
        client = AlpacaClient(settings=_settings())
        with pytest.raises(AlpacaCLIError) as exc_info:
            client.get_account()

    assert exc_info.value.returncode == 2
    assert exc_info.value.status == 401
    assert "unauthorized" in exc_info.value.stderr


@patch("alpaca_client.shutil.which", return_value="/usr/local/bin/alpaca")
def test_live_trade_flag_only_passed_through_when_explicitly_opted_in(mock_which):
    import os
    from alpaca_client import AlpacaClient

    runner = _fake_run(stdout="{}")
    with patch.dict(os.environ, {"ALPACA_LIVE_TRADE": "true"}), \
         patch("alpaca_client.subprocess.run", side_effect=runner):
        client = AlpacaClient(settings=_settings(i_understand_this_is_live=False))
        client.get_account()
        assert "ALPACA_LIVE_TRADE" not in runner.calls[0]["env"]

        client_live = AlpacaClient(settings=_settings(i_understand_this_is_live=True))
        client_live.get_account()
        assert runner.calls[1]["env"]["ALPACA_LIVE_TRADE"] == "true"


@patch("alpaca_client.shutil.which", return_value="/usr/local/bin/alpaca")
def test_option_chain_snapshot_paginates_on_next_page_token(mock_which):
    from alpaca_client import AlpacaClient
    responses = [
        json.dumps({"snapshots": {"SPY240119C00580000": {"impliedVolatility": 0.2}}, "next_page_token": "tok1"}),
        json.dumps({"snapshots": {"SPY240119P00580000": {"impliedVolatility": 0.21}}}),
    ]

    def _runner(args, input=None, capture_output=None, text=None, timeout=None, env=None):
        result = MagicMock()
        result.stdout, result.stderr, result.returncode = responses.pop(0), "", 0
        return result

    with patch("alpaca_client.subprocess.run", side_effect=_runner):
        client = AlpacaClient(settings=_settings())
        snapshots = client.get_option_chain_snapshot("SPY")

    assert set(snapshots.keys()) == {"SPY240119C00580000", "SPY240119P00580000"}
