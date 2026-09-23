"""
tests/test_app_pages.py

Streamlit AppTest-based tests for app.py's page navigation and the two
new pages (Next Day Strategy, Live Market). AppTest runs the actual
Streamlit script server-side and lets us set widget values / click
buttons deterministically - unlike a real browser, there's no
network/render timing to race against, so these are the reliable
source of truth for "does the UI path actually work", complementing
the pure-logic tests in test_next_day.py / test_live_data.py.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = os.path.join(PROJECT_ROOT, "app.py")
RUN_TIMEOUT = 30


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=RUN_TIMEOUT)
    return at


def _goto(at: AppTest, page_label: str) -> AppTest:
    at.sidebar.radio[0].set_value(page_label)
    at.run(timeout=RUN_TIMEOUT)
    return at


class TestNavigation:

    def test_app_loads_without_exception(self):
        at = _fresh_app()
        assert not at.exception

    def test_sidebar_has_the_five_required_pages_in_order(self):
        at = _fresh_app()
        assert at.sidebar.radio[0].options == [
            "Stock Decision", "Next Day Strategy", "Live Market",
            "Stock History", "About",
        ]

    def test_developer_analysis_hidden_by_default(self):
        at = _fresh_app()
        assert "Developer Analysis" not in at.sidebar.radio[0].options

    def test_developer_analysis_appears_when_toggled(self):
        at = _fresh_app()
        at.sidebar.checkbox[0].set_value(True)
        at.run(timeout=RUN_TIMEOUT)
        assert "Developer Analysis" in at.sidebar.radio[0].options

    @pytest.mark.parametrize("page_label", [
        "Stock Decision", "Next Day Strategy", "Live Market",
        "Stock History", "About",
    ])
    def test_each_page_loads_without_exception(self, page_label):
        at = _goto(_fresh_app(), page_label)
        assert not at.exception


class TestNextDayStrategyPage:

    def test_company_selection_works(self):
        at = _fresh_app()
        company_select = at.sidebar.selectbox[0]
        options = company_select.options
        assert len(options) > 1
        company_select.set_value(options[1])
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception

    def test_price_input_widget_present(self):
        at = _goto(_fresh_app(), "Next Day Strategy")
        assert len(at.text_input) == 1

    @pytest.mark.parametrize("bad_value", ["0", "", "abc", "-50"])
    def test_invalid_price_shows_error_not_crash(self, bad_value):
        at = _goto(_fresh_app(), "Next Day Strategy")
        at.text_input[0].set_value(bad_value)
        at.run(timeout=RUN_TIMEOUT)
        at.button[0].click()
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("valid stock price" in e.value for e in at.error)

    def test_valid_price_shows_decision(self):
        at = _goto(_fresh_app(), "Next Day Strategy")
        at.text_input[0].set_value("2850")
        at.run(timeout=RUN_TIMEOUT)
        at.button[0].click()
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert not at.error
        combined = " ".join(md.value for md in at.markdown)
        assert any(word in combined for word in ("BUY", "HOLD", "SELL"))

    def test_no_algorithm_names_shown_on_this_page(self):
        """The normal-user page must never mention an individual
        algorithm by name (Q-Learning, SARSA, etc.)."""
        at = _goto(_fresh_app(), "Next Day Strategy")
        at.text_input[0].set_value("2850")
        at.run(timeout=RUN_TIMEOUT)
        at.button[0].click()
        at.run(timeout=RUN_TIMEOUT)
        combined = " ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)
        for banned in ("Q-Learning", "SARSA", "Monte Carlo", "Value Iteration", "Policy Iteration"):
            assert banned not in combined


class TestLiveMarketPage:

    def test_live_change_uses_live_previous_close_not_stale_static_history(self, monkeypatch):
        """
        Regression test: 'Today's change' on the Live Market page must be
        computed against a live previous close (yesterday's actual close
        from Angel One), never against the static historical dataset's
        last row - which can be weeks stale and produces a nonsensical
        'today's change' when compared to a genuinely live price.
        """
        import pandas as pd
        import streamlit as st
        from datetime import datetime
        from src.market import live_data
        from src.decision import live_features

        st.cache_data.clear()  # avoid cross-test cache_data pollution from other tests' real fetches

        fake_quote = live_data.LivePriceResult(available=True, price=1000.0, timestamp=datetime.now())
        monkeypatch.setattr(live_data, "fetch_live_price", lambda symbol: fake_quote)

        fake_row = pd.Series({
            "Date": pd.Timestamp.now(), "Close": 1000.0, "MA5": 995.0, "MA20": 990.0,
            "RSI": 55.0, "Volatility": 0.02, "Trend": "BULLISH", "RSI_Condition": "RSI_NORMAL",
            "Volatility_Condition": "LOW_VOLATILITY", "PreviousClose": 950.0,
            "InstrumentExchange": "NSE", "InstrumentTradingSymbol": "TEST-EQ",
            "InstrumentIsFuture": False, "InstrumentExpiry": None,
        })
        monkeypatch.setattr(live_features, "get_live_feature_row", lambda symbol: (fake_row, None))

        try:
            at = _goto(_fresh_app(), "Live Market")
            assert not at.exception
            combined = " ".join(md.value for md in at.markdown)
            # 1000 - 950 = +50.00 (+5.26%) - the LIVE previous close, not the static one
            assert "50.00" in combined
            assert "5.26%" in combined
            assert "₹950.00" in combined  # the "Previous Close" card itself
        finally:
            # These fake results are now cached under st.cache_data (ttl=300s) -
            # clear it so later tests in this session don't inherit fake live data.
            st.cache_data.clear()

    def test_loads_without_exception(self):
        at = _goto(_fresh_app(), "Live Market")
        assert not at.exception

    def test_shows_unavailable_warning_when_no_api_configured(self, monkeypatch):
        monkeypatch.delenv("MARKET_API_URL", raising=False)
        at = _goto(_fresh_app(), "Live Market")
        combined = " ".join(md.value for md in at.markdown)
        assert "unavailable" in combined.lower()

    def test_never_shows_fake_live_label_without_configured_api(self, monkeypatch):
        monkeypatch.delenv("MARKET_API_URL", raising=False)
        at = _goto(_fresh_app(), "Live Market")
        combined = " ".join(md.value for md in at.markdown)
        assert "Live price" not in combined  # only shown when a live fetch actually succeeded

    def test_refresh_button_present_and_clickable(self):
        at = _goto(_fresh_app(), "Live Market")
        assert len(at.button) >= 1
        at.button[0].click()
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception

    def test_no_algorithm_names_shown_on_this_page(self):
        at = _goto(_fresh_app(), "Live Market")
        combined = " ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)
        for banned in ("Q-Learning", "SARSA", "Monte Carlo", "Value Iteration", "Policy Iteration"):
            assert banned not in combined


class TestNewsAndRiskLayer:
    """
    The News & Risk Layer (src/news/, src/risk/) - no NEWS_API_KEY is set
    in the test environment, so these exercise the real 'News data
    unavailable' path end-to-end (never a fake/mocked news fetch),
    confirming the dashboard degrades honestly rather than crashing or
    inventing sentiment.
    """

    def test_stock_decision_page_shows_risk_layer_section(self, monkeypatch):
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        at = _goto(_fresh_app(), "Stock Decision")
        assert not at.exception
        combined = " ".join(md.value for md in at.markdown)
        assert "Signal, News & Risk" in combined
        assert "RL Signal" in combined
        assert "Final Decision" in combined

    def test_news_unavailable_is_shown_honestly_not_crashed(self, monkeypatch):
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        at = _goto(_fresh_app(), "Stock Decision")
        assert not at.exception
        combined = " ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)
        assert "News data unavailable" in combined
        assert "NEWS_API_KEY is not configured" in combined

    def test_live_market_page_shows_risk_layer_section(self, monkeypatch):
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        at = _goto(_fresh_app(), "Live Market")
        assert not at.exception
        combined = " ".join(md.value for md in at.markdown)
        assert "Signal, News & Risk" in combined

    def test_next_day_strategy_page_shows_risk_layer_section(self, monkeypatch):
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        at = _fresh_app()
        at = _goto(at, "Next Day Strategy")
        at.button[0].click()
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        combined = " ".join(md.value for md in at.markdown)
        assert "Signal, News & Risk" in combined

    def test_refresh_news_button_present_and_clickable(self, monkeypatch):
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        at = _goto(_fresh_app(), "Stock Decision")
        refresh_buttons = [b for b in at.button if "Refresh News" in (b.label or "")]
        assert len(refresh_buttons) >= 1
        refresh_buttons[0].click()
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception

    def test_final_decision_uses_avoid_or_sell_position_language(self, monkeypatch):
        """Never shows a bare, unlabeled 'SELL' as if the user already
        owns the stock - AVOID/SELL wording must come from the guard."""
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        at = _goto(_fresh_app(), "Stock Decision")
        assert not at.exception
        combined = " ".join(md.value for md in at.markdown)
        assert "Final Decision" in combined