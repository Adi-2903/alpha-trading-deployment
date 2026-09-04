"""
tests/test_alpaca_client_cli.py -- Tests for alpaca_client.py's REST transport.

These mock `requests.request` entirely -- no real network call or Alpaca
credentials needed. What's verified:
  - AlpacaClient sends correct HTTP method, URL, and headers
  - Pagination works (next_page_token / page_token loop)
  - Non-2xx responses raise AlpacaAPIError with status + body
  - get_account / get_orders / submit_order / get_option_chain_snapshot
    all route to the right endpoints
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


def _mock_response(body: dict | list | str, status_code: int = 200):
    """Build a fake requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = (200 <= status_code < 300)
    raw = json.dumps(body) if not isinstance(body, str) else body
    resp.text = raw
    resp.content = raw.encode()
    resp.json = lambda: json.loads(raw)
    return resp


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_client_init_succeeds_with_valid_credentials():
    from alpaca_client import AlpacaClient
    client = AlpacaClient(settings=_settings())
    assert client.settings.api_key == "PKtest"


def test_client_init_raises_without_credentials():
    from alpaca_client import AlpacaClient
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        AlpacaClient(settings=_settings(api_key="", api_secret=""))


# ---------------------------------------------------------------------------
# GET /v2/account
# ---------------------------------------------------------------------------

def test_get_account_calls_correct_endpoint():
    from alpaca_client import AlpacaClient
    resp = _mock_response({"equity": "100000", "cash": "50000"})
    with patch("requests.request", return_value=resp) as mock_req:
        client = AlpacaClient(settings=_settings())
        account = client.get_account()

    assert account["equity"] == "100000"
    call = mock_req.call_args
    assert call[0][0] == "GET"
    assert "/v2/account" in call[0][1]
    # credentials go in headers, never in URL
    headers = call[1]["headers"]
    assert headers["APCA-API-KEY-ID"] == "PKtest"
    assert headers["APCA-API-SECRET-KEY"] == "secrettest"


# ---------------------------------------------------------------------------
# POST /v2/orders
# ---------------------------------------------------------------------------

def test_submit_order_posts_json_body():
    from alpaca_client import AlpacaClient
    resp = _mock_response({"id": "abc123", "status": "accepted"})
    payload = {"symbol": "AAPL", "qty": "1", "side": "buy", "type": "market", "time_in_force": "day"}
    with patch("requests.request", return_value=resp) as mock_req:
        client = AlpacaClient(settings=_settings())
        result = client.submit_order(payload)

    assert result["status"] == "accepted"
    call = mock_req.call_args
    assert call[0][0] == "POST"
    assert "/v2/orders" in call[0][1]
    assert call[1]["json"] == payload


# ---------------------------------------------------------------------------
# GET /v2/orders
# ---------------------------------------------------------------------------

def test_get_orders_builds_query_string_from_params():
    from alpaca_client import AlpacaClient
    resp = _mock_response([])
    with patch("requests.request", return_value=resp) as mock_req:
        client = AlpacaClient(settings=_settings())
        client.get_orders(status="open", limit=50)

    call = mock_req.call_args
    url = call[0][1]
    assert "/v2/orders" in url
    assert "status=open" in url
    assert "limit=50" in url


# ---------------------------------------------------------------------------
# Non-2xx → AlpacaAPIError
# ---------------------------------------------------------------------------

def test_nonzero_status_raises_alpaca_api_error():
    from alpaca_client import AlpacaAPIError, AlpacaClient
    resp = _mock_response({"message": "unauthorized"}, status_code=401)
    with patch("requests.request", return_value=resp):
        client = AlpacaClient(settings=_settings())
        with pytest.raises(AlpacaAPIError) as exc_info:
            client.get_account()

    assert exc_info.value.status == 401
    assert "unauthorized" in exc_info.value.body


# ---------------------------------------------------------------------------
# Pagination: get_option_chain_snapshot
# ---------------------------------------------------------------------------

def test_option_chain_snapshot_paginates_on_next_page_token():
    from alpaca_client import AlpacaClient

    page1 = _mock_response({
        "snapshots": {"SPY240119C00580000": {"impliedVolatility": 0.2}},
        "next_page_token": "tok1",
    })
    page2 = _mock_response({
        "snapshots": {"SPY240119P00580000": {"impliedVolatility": 0.21}},
    })

    with patch("requests.request", side_effect=[page1, page2]):
        client = AlpacaClient(settings=_settings())
        snapshots = client.get_option_chain_snapshot("SPY")

    assert set(snapshots.keys()) == {"SPY240119C00580000", "SPY240119P00580000"}


# ---------------------------------------------------------------------------
# Pagination: get_option_contracts
# ---------------------------------------------------------------------------

def test_option_contracts_paginates_on_page_token():
    from alpaca_client import AlpacaClient

    page1 = _mock_response({
        "option_contracts": [{"symbol": "SPY240119C00580000"}],
        "page_token": "tok2",
    })
    page2 = _mock_response({
        "option_contracts": [{"symbol": "SPY240119P00580000"}],
    })

    with patch("requests.request", side_effect=[page1, page2]):
        client = AlpacaClient(settings=_settings())
        contracts = client.get_option_contracts("SPY")

    assert len(contracts) == 2
    assert contracts[0]["symbol"] == "SPY240119C00580000"
    assert contracts[1]["symbol"] == "SPY240119P00580000"
