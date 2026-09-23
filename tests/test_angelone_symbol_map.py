"""
tests/test_angelone_symbol_map.py

Tests for src/market/angelone_symbol_map.py - mostly guarding the
safety property that matters most here: codes that are deliberately
left unmapped (futures series, unverified underlyings) must stay
unmapped and report why, rather than someone accidentally wiring in a
default that substitutes the wrong instrument's price.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.company_map import CODE_MAP  # noqa: E402
from src.market.angelone_symbol_map import SYMBOL_MAP, get_broker_symbol  # noqa: E402


def test_known_equity_code_maps_to_nse():
    mapping, err = get_broker_symbol("RELI")
    assert err is None
    assert mapping == ("NSE", "RELIANCE-EQ")


def test_futures_series_codes_map_to_dynamic_front_month_marker():
    expected = {
        "ADELc1_NS": "ADANIENT",
        "TCSc1_NS": "TCS",
        "SBIc1_NS": "SBIN",
    }
    for code, underlying in expected.items():
        mapping, err = get_broker_symbol(code)
        assert err is None
        assert mapping == ("NFO_FUT", underlying)


def test_unverified_codes_are_left_unmapped():
    for code in ("INGL", "MAXE"):
        mapping, err = get_broker_symbol(code)
        assert mapping is None


def test_unknown_code_reports_unknown_not_a_silent_default():
    mapping, err = get_broker_symbol("NOT_A_REAL_CODE")
    assert mapping is None
    assert "no Angel One symbol mapping" in err


def test_every_company_map_code_is_at_least_acknowledged():
    """
    Every code company_map.py knows about should appear in SYMBOL_MAP
    (mapped or explicitly None) so a new/renamed code can't silently
    fall through with a confusing "unknown code" message instead of a
    clear "needs mapping" one.
    """
    missing = [code for code in CODE_MAP if code not in SYMBOL_MAP]
    assert missing == [], f"These company_map.py codes have no entry (mapped or None) in angelone_symbol_map.py: {missing}"
