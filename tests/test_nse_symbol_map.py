"""
tests/test_nse_symbol_map.py

Tests for src/market/nse_symbol_map.py - the plain-ticker resolver
used by the generic live-price provider (see live_data.py's bug-fix
comment: sending the raw internal code straight to a real provider
only coincidentally worked for 3 of this project's 50 codes).
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.market.nse_symbol_map import get_ticker  # noqa: E402


def test_equity_code_resolves_to_plain_ticker_no_eq_suffix():
    ticker, err = get_ticker("RELI")
    assert err is None
    assert ticker == "RELIANCE"


def test_futures_code_resolves_to_underlying_ticker():
    ticker, err = get_ticker("TCSc1_NS")
    assert err is None
    assert ticker == "TCS"


def test_internal_code_is_not_returned_as_the_ticker():
    """Regression guard for the original bug: the internal project
    code must never be silently used as if it were the real ticker."""
    ticker, err = get_ticker("APLH")
    assert err is None
    assert ticker != "APLH"
    assert ticker == "APOLLOHOSP"


def test_unverified_code_stays_unmapped():
    for code in ("INGL", "MAXE"):
        ticker, err = get_ticker(code)
        assert ticker is None
        assert err is not None


def test_unknown_code_reports_unknown():
    ticker, err = get_ticker("NOT_A_REAL_CODE")
    assert ticker is None
    assert "unknown internal code" in err


def test_tata_motors_resolves_to_renamed_tmpv_not_defunct_tatamotors():
    """Regression test: Tata Motors Ltd was demerged and renamed Tata
    Motors Passenger Vehicles Ltd (NSE symbol TMPV) in Oct 2025; the old
    TATAMOTORS symbol no longer exists on NSE. Both TAMO (equity) and
    TAMOc1_NS (futures) must resolve to TMPV, not the defunct symbol."""
    ticker, err = get_ticker("TAMO")
    assert err is None
    assert ticker == "TMPV"

    ticker, err = get_ticker("TAMOc1_NS")
    assert err is None
    assert ticker == "TMPV"
