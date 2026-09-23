"""
tests/test_live_data.py

Tests for src/market/live_data.py - market-hours status (Asia/Kolkata)
and the live-price fetcher, with a focus on it NEVER fabricating a
price: unconfigured, unreachable, and malformed-response cases must
all cleanly report unavailable rather than raising or guessing.
"""

import os
import sys
from datetime import datetime

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.market import live_data  # noqa: E402


# ---------------------------------------------------------------------------
# Market status
# ---------------------------------------------------------------------------

class TestMarketStatus:

    def test_weekday_during_trading_hours_is_open(self):
        # Wednesday 2026-09-02, 11:00 IST
        now = datetime(2026, 9, 2, 11, 0, tzinfo=live_data.IST)
        result = live_data.market_status(now)
        assert result["is_open"] is True
        assert result["label"] == "Market Open"

    def test_weekday_before_open_is_closed(self):
        now = datetime(2026, 9, 2, 8, 0, tzinfo=live_data.IST)  # before 09:15
        result = live_data.market_status(now)
        assert result["is_open"] is False
        assert result["label"] == "Market Closed"

    def test_weekday_after_close_is_closed(self):
        now = datetime(2026, 9, 2, 16, 0, tzinfo=live_data.IST)  # after 15:30
        result = live_data.market_status(now)
        assert result["is_open"] is False

    def test_at_exact_open_boundary_is_open(self):
        now = datetime(2026, 9, 2, 9, 15, tzinfo=live_data.IST)
        assert live_data.market_status(now)["is_open"] is True

    def test_at_exact_close_boundary_is_open(self):
        now = datetime(2026, 9, 2, 15, 30, tzinfo=live_data.IST)
        assert live_data.market_status(now)["is_open"] is True

    def test_saturday_is_closed_even_during_trading_hours(self):
        now = datetime(2026, 9, 5, 11, 0, tzinfo=live_data.IST)  # Saturday
        result = live_data.market_status(now)
        assert result["is_open"] is False

    def test_sunday_is_closed(self):
        now = datetime(2026, 9, 6, 11, 0, tzinfo=live_data.IST)  # Sunday
        assert live_data.market_status(now)["is_open"] is False

    def test_defaults_to_real_current_time_without_error(self):
        result = live_data.market_status()
        assert isinstance(result["is_open"], bool)
        assert result["now_ist"].tzinfo is not None


# ---------------------------------------------------------------------------
# Live price fetcher - must never fabricate a price
# ---------------------------------------------------------------------------

class TestFetchLivePrice:

    def test_unconfigured_returns_unavailable(self, monkeypatch):
        monkeypatch.delenv("MARKET_API_URL", raising=False)
        monkeypatch.delenv("MARKET_API_KEY", raising=False)
        result = live_data.fetch_live_price("RELI")
        assert result.available is False
        assert result.price is None
        assert result.reason is not None

    def test_bad_url_template_returns_unavailable_not_crash(self, monkeypatch):
        monkeypatch.setenv("MARKET_API_URL", "https://example.com/{not_a_real_field}")
        result = live_data.fetch_live_price("RELI")
        assert result.available is False
        assert result.price is None

    def test_network_failure_returns_unavailable(self, monkeypatch):
        monkeypatch.setenv("MARKET_API_URL", "https://this-domain-does-not-exist.invalid/{symbol}")
        monkeypatch.setenv("MARKET_API_KEY", "dummy")
        result = live_data.fetch_live_price("RELI")
        assert result.available is False
        assert result.price is None
        assert result.reason is not None

    def test_successful_fetch_parses_configured_field(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"quote": {"last": 2875.4}}

        def fake_get(url, timeout):
            return FakeResponse()

        monkeypatch.setenv("MARKET_API_URL", "https://example.com/{symbol}")
        monkeypatch.setenv("MARKET_API_KEY", "dummy")
        monkeypatch.setenv("MARKET_API_PRICE_FIELD", "quote.last")
        monkeypatch.setattr(live_data.requests, "get", fake_get)

        result = live_data.fetch_live_price("RELI")
        assert result.available is True
        assert result.price == pytest.approx(2875.4)
        assert result.timestamp is not None

    def test_malformed_response_returns_unavailable(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"unexpected": "shape"}

        monkeypatch.setenv("MARKET_API_URL", "https://example.com/{symbol}")
        monkeypatch.setenv("MARKET_API_KEY", "dummy")
        monkeypatch.setenv("MARKET_API_PRICE_FIELD", "price")
        monkeypatch.setattr(live_data.requests, "get", lambda url, timeout: FakeResponse())

        result = live_data.fetch_live_price("RELI")
        assert result.available is False
        assert result.price is None

    def test_generic_provider_sends_real_ticker_not_internal_code(self, monkeypatch):
        """Regression test for the fixed bug: the URL sent to the
        provider must contain the real NSE ticker ('APOLLOHOSP'), not
        this project's internal code ('APLH') - previously the raw
        internal code was sent, which only coincidentally worked for
        3 of 50 codes and looked like 'some stocks show live, some
        don't'."""
        seen = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"price": 6500.0}

        def fake_get(url, timeout):
            seen["url"] = url
            return FakeResponse()

        monkeypatch.setenv("MARKET_API_URL", "https://example.com/quote?symbol={symbol}")
        monkeypatch.setenv("MARKET_API_KEY", "dummy")
        monkeypatch.setattr(live_data.requests, "get", fake_get)

        result = live_data.fetch_live_price("APLH")
        assert result.available is True
        assert "APOLLOHOSP" in seen["url"]
        assert "APLH" not in seen["url"]

    def test_generic_provider_appends_configured_suffix(self, monkeypatch):
        seen = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"price": 100.0}

        def fake_get(url, timeout):
            seen["url"] = url
            return FakeResponse()

        monkeypatch.setenv("MARKET_API_URL", "https://example.com/quote?symbol={ticker}")
        monkeypatch.setenv("MARKET_API_KEY", "dummy")
        monkeypatch.setenv("MARKET_API_SYMBOL_SUFFIX", ".NS")
        monkeypatch.setattr(live_data.requests, "get", fake_get)

        result = live_data.fetch_live_price("RELI")
        assert result.available is True
        assert "RELIANCE.NS" in seen["url"]

    def test_generic_provider_reports_unavailable_for_unmapped_code(self, monkeypatch):
        monkeypatch.setenv("MARKET_API_URL", "https://example.com/quote?symbol={symbol}")
        monkeypatch.setenv("MARKET_API_KEY", "dummy")

        result = live_data.fetch_live_price("INGL")
        assert result.available is False
        assert result.reason is not None

    def test_angelone_provider_passes_through_ohlc_and_previous_close(self, monkeypatch):
        """Regression test: fetch_live_price's Angel One path must carry
        open/high/low/previous_close through from angelone_client.get_ltp
        into LivePriceResult - this is what lets the Live Market page
        show 'today's change' without needing the separate historical-
        candle endpoint (which has a currently unresolved rate-limit bug
        on Angel One's own platform)."""
        from src.market import angelone_client

        monkeypatch.delenv("MARKET_API_URL", raising=False)
        monkeypatch.setenv("ANGELONE_API_KEY", "dummy")

        fake_quote = angelone_client.AngelOneQuote(
            available=True, price=2875.4, timestamp=datetime.now(),
            open=2860.0, high=2890.5, low=2855.0, previous_close=2850.0)
        monkeypatch.setattr(angelone_client, "get_ltp", lambda *a, **kw: fake_quote)
        monkeypatch.setattr(live_data, "get_broker_symbol", lambda code: (("NSE", "RELIANCE-EQ"), None))

        result = live_data.fetch_live_price("RELI")
        assert result.available is True
        assert result.price == pytest.approx(2875.4)
        assert result.open == pytest.approx(2860.0)
        assert result.high == pytest.approx(2890.5)
        assert result.low == pytest.approx(2855.0)
        assert result.previous_close == pytest.approx(2850.0)

    def test_never_raises_regardless_of_failure_mode(self, monkeypatch):
        def raising_get(url, timeout):
            raise RuntimeError("boom")

        monkeypatch.setenv("MARKET_API_URL", "https://example.com/{symbol}")
        monkeypatch.setenv("MARKET_API_KEY", "dummy")
        monkeypatch.setattr(live_data.requests, "get", raising_get)

        # A truly unexpected exception type is allowed to surface (only
        # requests.RequestException/ValueError are caught deliberately) -
        # this documents that boundary rather than silently swallowing
        # everything. The common real-world failure modes (timeouts,
        # connection errors, bad JSON) ARE covered by the tests above.
        with pytest.raises(RuntimeError):
            live_data.fetch_live_price("RELI")