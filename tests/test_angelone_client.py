"""
tests/test_angelone_client.py

Tests for src/market/angelone_client.py - login/session handling and
LTP fetching, with a focus on it NEVER fabricating a price: missing
credentials, bad TOTP secrets, network failures, auth errors, and
unknown symbols must all cleanly report unavailable rather than
raising or guessing. All Angel One HTTP calls are mocked - these tests
never make a real network request.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.market import angelone_client  # noqa: E402


VALID_ENV = {
    "ANGELONE_API_KEY": "test_api_key",
    "ANGELONE_CLIENT_CODE": "A123456",
    "ANGELONE_PASSWORD": "1234",
    # A syntactically-valid base32 TOTP secret (not a real account secret).
    "ANGELONE_TOTP_SECRET": "JBSWY3DPEHPK3PXP",
}


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Every test starts with no cached session and no memoized scrip master."""
    angelone_client.reset_session_cache()
    angelone_client._scrip_master_memory_cache = None
    for key in VALID_ENV:
        monkeypatch.delenv(key, raising=False)
    yield
    angelone_client.reset_session_cache()
    angelone_client._scrip_master_memory_cache = None


def _set_valid_env(monkeypatch):
    for key, value in VALID_ENV.items():
        monkeypatch.setenv(key, value)


class FakeResponse:
    def __init__(self, json_body, raise_exc=None):
        self._json_body = json_body
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._json_body


LOGIN_SUCCESS_BODY = {
    "status": True,
    "message": "SUCCESS",
    "data": {"jwtToken": "fake.jwt.token", "refreshToken": "fake_refresh", "feedToken": "fake_feed"},
}

LTP_SUCCESS_BODY = {
    "status": True,
    "message": "SUCCESS",
    "data": {"exchange": "NSE", "tradingsymbol": "RELIANCE-EQ", "symboltoken": "2885", "ltp": 2875.4,
              "open": 2860.0, "high": 2890.5, "low": 2855.0, "close": 2850.0},
}


# ---------------------------------------------------------------------------
# Credentials / configuration
# ---------------------------------------------------------------------------

class TestCredentials:

    def test_missing_credentials_returns_unavailable(self):
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is False
        assert result.price is None
        assert "not configured" in result.reason

    def test_partial_credentials_returns_unavailable(self, monkeypatch):
        monkeypatch.setenv("ANGELONE_API_KEY", "test_api_key")
        # client code / password / totp secret left unset
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is False
        assert "not configured" in result.reason


# ---------------------------------------------------------------------------
# Login / session
# ---------------------------------------------------------------------------

class TestLogin:

    def test_invalid_totp_secret_returns_unavailable(self, monkeypatch):
        _set_valid_env(monkeypatch)
        monkeypatch.setenv("ANGELONE_TOTP_SECRET", "not valid base32!!")
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is False
        assert "TOTP secret" in result.reason

    def test_login_network_failure_returns_unavailable(self, monkeypatch):
        _set_valid_env(monkeypatch)

        def fake_post(url, json, headers, timeout):
            raise angelone_client.requests.ConnectionError("boom")

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is False
        assert "login" in result.reason.lower()

    def test_login_rejected_returns_unavailable(self, monkeypatch):
        _set_valid_env(monkeypatch)

        def fake_post(url, json, headers, timeout):
            return FakeResponse({"status": False, "message": "Invalid totp", "errorcode": "AB1050"})

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is False
        assert "Invalid totp" in result.reason

    def test_successful_login_then_ltp_fetch(self, monkeypatch):
        _set_valid_env(monkeypatch)
        calls = {"login": 0, "ltp": 0}

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                calls["login"] += 1
                assert json["clientcode"] == "A123456"
                assert len(json["totp"]) == 6  # pyotp generates a 6-digit code
                return FakeResponse(LOGIN_SUCCESS_BODY)
            if url.endswith(angelone_client.LTP_PATH):
                calls["ltp"] += 1
                assert headers["Authorization"] == "Bearer fake.jwt.token"
                return FakeResponse(LTP_SUCCESS_BODY)
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")

        assert result.available is True
        assert result.price == pytest.approx(2875.4)
        assert calls["login"] == 1
        assert calls["ltp"] == 1

    def test_session_is_reused_across_calls(self, monkeypatch):
        _set_valid_env(monkeypatch)
        calls = {"login": 0}

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                calls["login"] += 1
                return FakeResponse(LOGIN_SUCCESS_BODY)
            return FakeResponse(LTP_SUCCESS_BODY)

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")

        assert calls["login"] == 1  # second call reused the cached session

    def test_auth_error_on_ltp_triggers_one_relogin_retry(self, monkeypatch):
        _set_valid_env(monkeypatch)
        calls = {"login": 0, "ltp": 0}

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                calls["login"] += 1
                return FakeResponse(LOGIN_SUCCESS_BODY)
            if url.endswith(angelone_client.LTP_PATH):
                calls["ltp"] += 1
                if calls["ltp"] == 1:
                    return FakeResponse({"status": False, "message": "Invalid Token", "errorcode": "AG8001"})
                return FakeResponse(LTP_SUCCESS_BODY)
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")

        assert result.available is True
        assert calls["login"] == 2  # initial + one re-login after the auth error
        assert calls["ltp"] == 2


# ---------------------------------------------------------------------------
# Scrip master / symbol token lookup
# ---------------------------------------------------------------------------

class TestScripMasterLookup:

    def test_lookup_finds_matching_token(self, monkeypatch, tmp_path):
        cache_path = tmp_path / "scrip_master.json"
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", cache_path)

        scrips = [
            {"token": "2885", "symbol": "RELIANCE-EQ", "exch_seg": "NSE"},
            {"token": "1594", "symbol": "INFY-EQ", "exch_seg": "NSE"},
        ]

        def fake_get(url, timeout):
            return FakeResponse(scrips)

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        token, err = angelone_client.lookup_symbol_token("NSE", "RELIANCE-EQ")
        assert token == "2885"
        assert err is None

    def test_lookup_unknown_symbol_returns_reason(self, monkeypatch, tmp_path):
        cache_path = tmp_path / "scrip_master.json"
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", cache_path)

        def fake_get(url, timeout):
            return FakeResponse([{"token": "2885", "symbol": "RELIANCE-EQ", "exch_seg": "NSE"}])

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        token, err = angelone_client.lookup_symbol_token("NSE", "DOESNOTEXIST-EQ")
        assert token is None
        assert "not found" in err

    def test_scrip_master_download_failure_returns_reason(self, monkeypatch, tmp_path):
        cache_path = tmp_path / "scrip_master.json"
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", cache_path)

        def fake_get(url, timeout):
            raise angelone_client.requests.ConnectionError("boom")

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        token, err = angelone_client.lookup_symbol_token("NSE", "RELIANCE-EQ")
        assert token is None
        assert "instrument list" in err

    def test_get_ltp_uses_lookup_when_no_token_given(self, monkeypatch, tmp_path):
        _set_valid_env(monkeypatch)
        cache_path = tmp_path / "scrip_master.json"
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", cache_path)

        def fake_get(url, timeout):
            return FakeResponse([{"token": "2885", "symbol": "RELIANCE-EQ", "exch_seg": "NSE"}])

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            assert json["symboltoken"] == "2885"  # looked up automatically
            return FakeResponse(LTP_SUCCESS_BODY)

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        monkeypatch.setattr(angelone_client.requests, "post", fake_post)

        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ")
        assert result.available is True
        assert result.price == pytest.approx(2875.4)


# ---------------------------------------------------------------------------
# Futures - front-month contract resolution
# ---------------------------------------------------------------------------

class TestFrontMonthFutures:

    def _fake_scrips(self, offset_days_list):
        """Builds a fake scrip master with ADANIENT futures contracts at the given day offsets from today."""
        import datetime as dt
        base = dt.date.today()
        expiries = [base + dt.timedelta(days=d) for d in offset_days_list]
        rows = []
        for i, exp in enumerate(expiries):
            rows.append({
                "token": f"1000{i}",
                "symbol": f"ADANIENT{exp.strftime('%d%b%Y').upper()}FUT",
                "name": "ADANIENT",
                "expiry": exp.strftime("%d%b%Y").upper(),
                "instrumenttype": "FUTSTK",
                "exch_seg": "NFO",
            })
        # a decoy from a different underlying, must never be picked
        rows.append({
            "token": "99999",
            "symbol": "TCS30SEP2026FUT",
            "name": "TCS",
            "expiry": (base + dt.timedelta(days=5)).strftime("%d%b%Y").upper(),
            "instrumenttype": "FUTSTK",
            "exch_seg": "NFO",
        })
        return rows

    def test_picks_nearest_not_yet_expired_contract(self, monkeypatch, tmp_path):
        cache_path = tmp_path / "scrip_master.json"
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", cache_path)

        # one already-expired (-10 days, must be skipped), two future ones
        scrips = self._fake_scrips([-10, 20, 50])

        def fake_get(url, timeout):
            return FakeResponse(scrips)

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        contract, err = angelone_client.resolve_front_month_future("ADANIENT")

        assert err is None
        assert contract is not None
        # the +20-day contract should win, not the expired one or the +50-day one
        expected_symbol = scrips[1]["symbol"]
        assert contract["tradingsymbol"] == expected_symbol
        assert contract["token"] == scrips[1]["token"]

    def test_unknown_underlying_returns_reason(self, monkeypatch, tmp_path):
        cache_path = tmp_path / "scrip_master.json"
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", cache_path)

        def fake_get(url, timeout):
            return FakeResponse(self._fake_scrips([20]))

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        contract, err = angelone_client.resolve_front_month_future("NOT_A_REAL_UNDERLYING")

        assert contract is None
        assert "No live NFO futures contract" in err

    def test_get_ltp_future_end_to_end(self, monkeypatch, tmp_path):
        _set_valid_env(monkeypatch)
        cache_path = tmp_path / "scrip_master.json"
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", cache_path)

        scrips = self._fake_scrips([20])
        front_month_symbol = scrips[0]["symbol"]
        front_month_token = scrips[0]["token"]

        def fake_get(url, timeout):
            return FakeResponse(scrips)

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            assert json["exchange"] == "NFO"
            assert json["tradingsymbol"] == front_month_symbol
            assert json["symboltoken"] == front_month_token
            return FakeResponse({
                "status": True,
                "data": {"exchange": "NFO", "tradingsymbol": front_month_symbol,
                          "symboltoken": front_month_token, "ltp": 3120.5},
            })

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        monkeypatch.setattr(angelone_client.requests, "post", fake_post)

        result = angelone_client.get_ltp_future("ADANIENT")
        assert result.available is True
        assert result.price == pytest.approx(3120.5)

    def test_expiry_parsing_handles_multiple_formats(self):
        assert angelone_client._parse_expiry("27JUL2023") is not None
        assert angelone_client._parse_expiry("25JAN24") is not None
        assert angelone_client._parse_expiry("2023-07-27") is not None
        assert angelone_client._parse_expiry("") is None
        assert angelone_client._parse_expiry(None) is None
        assert angelone_client._parse_expiry("not-a-date") is None


# ---------------------------------------------------------------------------
# LTP - malformed responses never fabricate a price
# ---------------------------------------------------------------------------

class TestLtpMalformedResponses:

    def test_ltp_network_failure_returns_unavailable(self, monkeypatch):
        _set_valid_env(monkeypatch)

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            raise angelone_client.requests.ConnectionError("boom")

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is False
        assert result.price is None

    def test_ltp_missing_price_field_returns_unavailable(self, monkeypatch):
        _set_valid_env(monkeypatch)

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            return FakeResponse({"status": True, "data": {"exchange": "NSE"}})  # no "ltp"

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is False
        assert result.price is None

    def test_ltp_response_surfaces_ohlc_and_previous_close(self, monkeypatch):
        """Regression test: Angel One's getLtpData response already
        includes open/high/low/close (previous close) alongside ltp in
        the SAME call - these must be parsed and returned so the app
        can show 'today's change' / previous close without depending on
        the separate historical-candle endpoint (which has a long-
        standing, currently unresolved false-positive rate-limit bug on
        Angel One's own platform)."""
        _set_valid_env(monkeypatch)

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            return FakeResponse(LTP_SUCCESS_BODY)

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is True
        assert result.price == pytest.approx(2875.4)
        assert result.open == pytest.approx(2860.0)
        assert result.high == pytest.approx(2890.5)
        assert result.low == pytest.approx(2855.0)
        assert result.previous_close == pytest.approx(2850.0)

    def test_ltp_response_missing_ohlc_fields_degrades_gracefully(self, monkeypatch):
        """If a provider response only has 'ltp' (no OHLC fields), the
        price must still resolve and the OHLC fields must be None
        rather than crashing or fabricating values."""
        _set_valid_env(monkeypatch)

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            return FakeResponse({"status": True, "data": {"exchange": "NSE", "ltp": 100.0}})

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is True
        assert result.price == pytest.approx(100.0)
        assert result.open is None
        assert result.high is None
        assert result.low is None
        assert result.previous_close is None


class TestTransientNetworkRetry:
    """
    Regression tests for the added retry-once-on-transient-failure
    behavior (fixes a one-off timeout/connection blip - especially
    common on the historical-candle endpoint - looking identical to a
    persistently broken stock)."""

    def test_ltp_recovers_after_one_transient_timeout(self, monkeypatch):
        _set_valid_env(monkeypatch)
        monkeypatch.setattr(angelone_client.time, "sleep", lambda *_: None)
        calls = {"n": 0}

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            calls["n"] += 1
            if calls["n"] == 1:
                raise angelone_client.requests.Timeout("slow")
            return FakeResponse(LTP_SUCCESS_BODY)

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is True
        assert result.price == pytest.approx(2875.4)
        assert calls["n"] == 2  # one failed attempt, one successful retry

    def test_historical_candles_recovers_after_transient_non_json_response(self, monkeypatch):
        _set_valid_env(monkeypatch)
        monkeypatch.setattr(angelone_client.time, "sleep", lambda *_: None)
        calls = {"n": 0}
        candle_body = {"status": True, "data": [["2026-09-01T00:00:00+05:30", 100, 105, 99, 102, 1000]]}

        class BadJsonResponse:
            status_code = 429
            text = "Rate limit exceeded, please retry after some time"

            def json(self):
                raise ValueError("not json")

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            calls["n"] += 1
            if calls["n"] == 1:
                return BadJsonResponse()
            return FakeResponse(candle_body)

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        from datetime import datetime
        candles, reason = angelone_client.get_historical_candles(
            "NSE", "2885", datetime(2026, 1, 1), datetime(2026, 9, 1))
        assert candles is not None
        assert reason is None
        assert calls["n"] == 2

    def test_persistent_failure_still_cleanly_reports_unavailable(self, monkeypatch):
        _set_valid_env(monkeypatch)
        monkeypatch.setattr(angelone_client.time, "sleep", lambda *_: None)

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            raise angelone_client.requests.ConnectionError("still down")

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        result = angelone_client.get_ltp("NSE", "RELIANCE-EQ", symboltoken="2885")
        assert result.available is False
        assert "connection error" in result.reason

    def test_non_json_failure_surfaces_status_code_and_body_snippet(self, monkeypatch):
        """Regression test: the reason string must show WHAT Angel One
        actually sent back (status code + a snippet of the body), not
        just a generic 'non-JSON response' - this is what lets someone
        tell a rate limit apart from an auth/subscription gateway page
        without needing to add more logging themselves."""
        _set_valid_env(monkeypatch)
        monkeypatch.setattr(angelone_client.time, "sleep", lambda *_: None)

        class RateLimitedResponse:
            status_code = 429
            text = "Too Many Requests - historical data quota exceeded for this API key"

            def json(self):
                raise ValueError("not json")

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            return RateLimitedResponse()

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        from datetime import datetime
        candles, reason = angelone_client.get_historical_candles(
            "NSE", "2885", datetime(2026, 1, 1), datetime(2026, 9, 1))
        assert candles is None
        assert "429" in reason
        assert "quota exceeded" in reason

    def test_angel_one_rate_limit_block_skips_pointless_retry(self, monkeypatch):
        """Regression test: Angel One's own '403 - exceeding access rate'
        response is a hard policy block, not a blip - retrying
        immediately is guaranteed to hit the same wall, so the code must
        return after the FIRST attempt (not burn the retry/backoff) and
        must clearly say this is a rate limit rather than a generic
        network failure."""
        _set_valid_env(monkeypatch)
        monkeypatch.setattr(angelone_client.time, "sleep", lambda *_: None)
        calls = {"n": 0}

        class RateLimitBlockedResponse:
            status_code = 403
            text = "Access denied because of exceeding access rate"

            def json(self):
                raise ValueError("not json")

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            calls["n"] += 1
            return RateLimitBlockedResponse()

        monkeypatch.setattr(angelone_client.requests, "post", fake_post)
        from datetime import datetime
        candles, reason = angelone_client.get_historical_candles(
            "NSE", "2885", datetime(2026, 1, 1), datetime(2026, 9, 1))
        assert candles is None
        assert calls["n"] == 1  # did NOT retry a doomed request
        assert "rate-limited" in reason.lower()
        assert "wait" in reason.lower()
