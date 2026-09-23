"""
tests/test_live_features.py

Tests for src/decision/live_features.py - computing TODAY's live
Trend/RSI/Volatility indicators from Angel One historical candles,
using the exact same formulas/thresholds as the static training
pipeline. All Angel One calls are mocked - no real account needed.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.decision import live_features  # noqa: E402
from src.market import angelone_client  # noqa: E402


VALID_ENV = {
    "ANGELONE_API_KEY": "test_api_key",
    "ANGELONE_CLIENT_CODE": "A123456",
    "ANGELONE_PASSWORD": "1234",
    "ANGELONE_TOTP_SECRET": "JBSWY3DPEHPK3PXP",
}

LOGIN_SUCCESS_BODY = {
    "status": True,
    "data": {"jwtToken": "fake.jwt.token", "refreshToken": "fake_refresh", "feedToken": "fake_feed"},
}


class FakeResponse:
    def __init__(self, json_body):
        self._json_body = json_body

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_body


def _set_valid_env(monkeypatch):
    for key, value in VALID_ENV.items():
        monkeypatch.setenv(key, value)


def _make_candles(n_days: int, start_price: float = 100.0, step: float = 0.5,
                   base_volume: float = 100000.0, volume_step: float = 0.0, high_low_pad: float = 0.0):
    """n_days ascending daily candles ending today, business days only."""
    end = datetime.now()
    dates = pd.bdate_range(end=end, periods=n_days)
    candles = []
    for i, d in enumerate(dates):
        close = start_price + i * step
        volume = base_volume + i * volume_step
        high, low = close + high_low_pad, close - high_low_pad
        candles.append([d.strftime("%Y-%m-%dT00:00:00+05:30"), close, high, low, close, volume])
    return candles


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    angelone_client.reset_session_cache()
    angelone_client._scrip_master_memory_cache = None
    live_features._threshold_cache = None
    for key in VALID_ENV:
        monkeypatch.delenv(key, raising=False)
    yield
    angelone_client.reset_session_cache()
    angelone_client._scrip_master_memory_cache = None
    live_features._threshold_cache = None


@pytest.fixture
def fake_threshold_file(tmp_path, monkeypatch):
    path = tmp_path / "volatility_thresholds.csv"
    pd.DataFrame([
        {"Company": "Reliance Industries", "Symbol": "RELI", "Volatility_Threshold_Train_Median": 0.02},
        {"Company": "Fake Unmapped Co", "Symbol": "NOT_A_REAL_CODE", "Volatility_Threshold_Train_Median": 0.02},
    ]).to_csv(path, index=False)
    monkeypatch.setattr(live_features, "VOLATILITY_THRESHOLDS_PATH", str(path))
    return path


class TestResolveInstrumentAndCredentials:

    def test_missing_credentials_returns_reason(self, fake_threshold_file):
        row, reason = live_features.get_live_feature_row("RELI")
        assert row is None
        assert reason is not None

    def test_unmapped_symbol_returns_reason(self, monkeypatch, fake_threshold_file):
        _set_valid_env(monkeypatch)
        row, reason = live_features.get_live_feature_row("NOT_A_REAL_CODE")
        assert row is None
        assert "no Angel One symbol mapping" in reason


class TestVolatilityThreshold:

    def test_missing_threshold_file_returns_reason(self, monkeypatch, tmp_path):
        _set_valid_env(monkeypatch)
        monkeypatch.setattr(live_features, "VOLATILITY_THRESHOLDS_PATH", str(tmp_path / "missing.csv"))
        row, reason = live_features.get_live_feature_row("RELI")
        assert row is None
        assert "not found" in reason

    def test_missing_symbol_in_threshold_file_returns_reason(self, monkeypatch, tmp_path):
        _set_valid_env(monkeypatch)
        path = tmp_path / "volatility_thresholds.csv"
        pd.DataFrame([
            {"Company": "Infosys", "Symbol": "INFY", "Volatility_Threshold_Train_Median": 0.02},
        ]).to_csv(path, index=False)
        monkeypatch.setattr(live_features, "VOLATILITY_THRESHOLDS_PATH", str(path))
        row, reason = live_features.get_live_feature_row("RELI")
        assert row is None
        assert "No trained volatility threshold" in reason


class TestLiveFeatureComputation:

    def test_insufficient_history_returns_reason(self, monkeypatch, fake_threshold_file, tmp_path):
        _set_valid_env(monkeypatch)
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", tmp_path / "scrip_master.json")

        def fake_get(url, timeout):
            return FakeResponse([{"token": "2885", "symbol": "RELIANCE-EQ", "exch_seg": "NSE"}])

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            if url.endswith(angelone_client.HISTORICAL_PATH):
                return FakeResponse({"status": True, "message": "SUCCESS", "data": _make_candles(5)})
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        monkeypatch.setattr(angelone_client.requests, "post", fake_post)

        row, reason = live_features.get_live_feature_row("RELI")
        assert row is None
        assert "Only 5 live trading day" in reason

    def test_successful_live_computation(self, monkeypatch, fake_threshold_file, tmp_path):
        _set_valid_env(monkeypatch)
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", tmp_path / "scrip_master.json")
        candles = _make_candles(40, volume_step=1000.0, high_low_pad=2.0)

        def fake_get(url, timeout):
            return FakeResponse([{"token": "2885", "symbol": "RELIANCE-EQ", "exch_seg": "NSE"}])

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            if url.endswith(angelone_client.HISTORICAL_PATH):
                assert json["exchange"] == "NSE"
                assert json["symboltoken"] == "2885"
                assert json["interval"] == "ONE_DAY"
                return FakeResponse({"status": True, "message": "SUCCESS", "data": candles})
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        monkeypatch.setattr(angelone_client.requests, "post", fake_post)

        row, reason = live_features.get_live_feature_row("RELI")
        assert reason is None
        assert row is not None
        # last candle close = 100 + 39*0.5 = 119.5, strictly increasing prices -> BULLISH, above MA20
        assert row["Close"] == pytest.approx(119.5)
        assert row["Trend"] == "BULLISH"
        assert not pd.isna(row["MA5"])
        assert not pd.isna(row["MA20"])
        # second-to-last close = 100 + 38*0.5 = 119.0
        assert row["PreviousClose"] == pytest.approx(119.0)

        # Today's High/Low/Volume come straight from the live candle series.
        assert row["High"] == pytest.approx(121.5)   # Close + pad(2.0)
        assert row["Low"] == pytest.approx(117.5)    # Close - pad(2.0)
        assert row["Volume"] == pytest.approx(100000.0 + 39 * 1000.0)
        assert row["PreviousVolume"] == pytest.approx(100000.0 + 38 * 1000.0)
        assert row["VolumeChangePct"] == pytest.approx(
            (row["Volume"] - row["PreviousVolume"]) / row["PreviousVolume"] * 100.0
        )

        # 40 trading days is enough for MACD's 34-day warm-up (slow=26 + signal=9 - 1).
        assert row["MACD_Available"] is True
        assert not pd.isna(row["MACD"])
        assert not pd.isna(row["MACD_Signal"])
        # Strictly rising, accelerating-then-linear closes -> fast EMA above slow EMA -> positive MACD.
        assert row["MACD"] > 0

    def test_macd_unavailable_with_insufficient_history_but_core_decision_still_works(
        self, monkeypatch, fake_threshold_file, tmp_path
    ):
        """
        25 trading days is enough for the CORE decision (MA20/RSI14/
        Volatility20 all need <= 20-21) but NOT enough for MACD's 34-day
        warm-up - the live decision must still succeed, just without MACD.
        """
        _set_valid_env(monkeypatch)
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", tmp_path / "scrip_master.json")
        candles = _make_candles(25)

        def fake_get(url, timeout):
            return FakeResponse([{"token": "2885", "symbol": "RELIANCE-EQ", "exch_seg": "NSE"}])

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            if url.endswith(angelone_client.HISTORICAL_PATH):
                return FakeResponse({"status": True, "message": "SUCCESS", "data": candles})
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        monkeypatch.setattr(angelone_client.requests, "post", fake_post)

        row, reason = live_features.get_live_feature_row("RELI")
        assert reason is None
        assert row is not None
        assert not pd.isna(row["MA20"])          # core decision inputs are fine
        assert row["MACD_Available"] is False     # but MACD hasn't warmed up yet
        assert pd.isna(row["MACD"])

    def test_historical_data_failure_returns_reason(self, monkeypatch, fake_threshold_file, tmp_path):
        _set_valid_env(monkeypatch)
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", tmp_path / "scrip_master.json")

        def fake_get(url, timeout):
            return FakeResponse([{"token": "2885", "symbol": "RELIANCE-EQ", "exch_seg": "NSE"}])

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            if url.endswith(angelone_client.HISTORICAL_PATH):
                raise angelone_client.requests.ConnectionError("boom")
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        monkeypatch.setattr(angelone_client.requests, "post", fake_post)

        row, reason = live_features.get_live_feature_row("RELI")
        assert row is None
        assert "historical-data" in reason.lower()


class TestFuturesUnderlyingResolution:

    def test_futures_code_resolves_via_front_month_then_fetches_candles(self, monkeypatch, tmp_path):
        _set_valid_env(monkeypatch)
        path = tmp_path / "volatility_thresholds.csv"
        pd.DataFrame([
            {"Company": "Adani Enterprises (Futures)", "Symbol": "ADELc1_NS", "Volatility_Threshold_Train_Median": 0.03},
        ]).to_csv(path, index=False)
        monkeypatch.setattr(live_features, "VOLATILITY_THRESHOLDS_PATH", str(path))
        monkeypatch.setattr(angelone_client, "SCRIP_MASTER_CACHE_PATH", tmp_path / "scrip_master.json")

        base = datetime.now().date()
        scrips = [{
            "token": "55555", "symbol": "ADANIENT25SEPFUT", "name": "ADANIENT",
            "expiry": (base + timedelta(days=20)).strftime("%d%b%Y").upper(),
            "instrumenttype": "FUTSTK", "exch_seg": "NFO",
        }]
        candles = _make_candles(40)

        def fake_get(url, timeout):
            return FakeResponse(scrips)

        def fake_post(url, json, headers, timeout):
            if url.endswith(angelone_client.LOGIN_PATH):
                return FakeResponse(LOGIN_SUCCESS_BODY)
            if url.endswith(angelone_client.HISTORICAL_PATH):
                assert json["exchange"] == "NFO"
                assert json["symboltoken"] == "55555"
                return FakeResponse({"status": True, "message": "SUCCESS", "data": candles})
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(angelone_client.requests, "get", fake_get)
        monkeypatch.setattr(angelone_client.requests, "post", fake_post)

        row, reason = live_features.get_live_feature_row("ADELc1_NS")
        assert reason is None
        assert row is not None
