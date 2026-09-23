"""
src/market/live_data.py

Dashboard update - "Live Market" page support.

TWO INDEPENDENT PIECES
-----------------------
1. `market_status()` - is the NSE session open right now? Pure
   date/time logic (Asia/Kolkata timezone, Mon-Fri, 09:15-15:30 IST).
   No network call, always available.

2. `fetch_live_price()` - the actual current traded price, from a
   live market-data API. THIS PROJECT DOES NOT SHIP WITH A CONFIGURED
   LIVE MARKET-DATA PROVIDER (Phase 1 explicitly excludes live APIs -
   see Sprint 4/5's "DO NOT add live API"), and this project's company
   codes (e.g. "RELI", "ADELc1_NS" - see src/data/company_map.py) are
   internal identifiers, not verified real-world exchange ticker
   symbols, so there is no safe default endpoint to hard-code even if
   we wanted to.

   Rather than faking a live price or pretending a specific vendor is
   wired up, this module supports two provider modes:

   (a) "generic" (default) - a GENERIC, provider-agnostic HTTP client:
       point it at any JSON quote endpoint via environment variables
       (see .env.example) and it will make a real HTTP request and
       parse a real response.
           MARKET_API_KEY          - your provider's API key
           MARKET_API_URL          - a URL template containing
                                      {symbol} (or, equivalently,
                                      {ticker}) and, if needed,
                                      {api_key}, e.g.:
                                      "https://api.example.com/v1/quote?symbol={symbol}&apikey={api_key}"
                                      {symbol}/{ticker} are resolved to
                                      the real plain NSE ticker (e.g.
                                      "RELIANCE", not this project's
                                      internal code "RELI") via
                                      src/market/nse_symbol_map.py
                                      before the request is made.
           MARKET_API_PRICE_FIELD  - dot-path into the JSON response
                                      for the current price, e.g.
                                      "price" or "quote.last"
                                      (default: "price")
           MARKET_API_SYMBOL_SUFFIX - optional, appended to the
                                      resolved ticker, e.g. ".NS" for
                                      providers that use Yahoo-style
                                      exchange suffixes (default: "")

   (b) "angelone" - Angel One's SmartAPI. See
       src/market/angelone_client.py (auth/session details) and
       src/market/angelone_symbol_map.py (internal code ->
       tradingsymbol mapping - READ ITS WARNINGS before trusting a
       price for a real trade). Set MARKET_PROVIDER=angelone plus the
       ANGELONE_* variables in .env.example. If MARKET_PROVIDER is
       unset but ANGELONE_API_KEY is present, Angel One is used
       automatically; otherwise the generic provider is used.

   Until one of these is configured, `fetch_live_price()` always and
   honestly reports the data as unavailable - this is the correct
   default behavior for this project, not a bug.

   `fetch_live_price()` only ever returns the current price and a
   fetch timestamp - it never invents Day High/Low/Volume from a field
   it can't confidently locate; the Live Market page shows those as
   "Not available" when this returns them as None, rather than
   guessing.
"""

import os
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from src.market import angelone_client
from src.market.angelone_symbol_map import get_broker_symbol
from src.market.nse_symbol_map import get_ticker

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
REQUEST_TIMEOUT_SECONDS = 5


def market_status(now: Optional[datetime] = None) -> dict:
    """
    Weekday + trading-hours check only (Section "MARKET HOURS") - does
    NOT account for exchange holidays, since this project has no
    holiday calendar data source. `now` is injectable for testing;
    defaults to the real current time in Asia/Kolkata.
    """
    now_ist = (now.astimezone(IST) if now.tzinfo else now.replace(tzinfo=IST)) if now else datetime.now(IST)
    is_weekday = now_ist.weekday() < 5  # Monday=0 ... Friday=4
    is_trading_hours = MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE
    is_open = is_weekday and is_trading_hours
    return {
        "is_open": is_open,
        "label": "Market Open" if is_open else "Market Closed",
        "now_ist": now_ist,
    }


def _resolve_field(payload: dict, dot_path: str):
    """Walk a dict via a dot-path like 'quote.last'. Returns None on any miss."""
    value = payload
    for part in dot_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


@dataclass
class LivePriceResult:
    available: bool
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    reason: Optional[str] = None
    # BUG FIX (2026-09): populated from Angel One's LTP response (which
    # already includes these alongside the price itself) so "today's
    # change" / previous close / today's high-low no longer require a
    # separate call to the historical-candle endpoint - see
    # AngelOneQuote in angelone_client.py for why that endpoint can't be
    # relied on. None for the generic provider (which doesn't offer
    # these fields) or if Angel One's response happened to omit them.
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    previous_close: Optional[float] = None


def _unavailable(reason: str) -> LivePriceResult:
    return LivePriceResult(available=False, reason=reason)


def fetch_live_price(symbol: str) -> LivePriceResult:
    """
    Dispatches to whichever provider is configured (see module
    docstring): "angelone" or the "generic" templated-URL client.
    Returns an honest "unavailable" result (never a fake price) if
    unconfigured, unreachable, or the response can't be parsed - the
    caller (the Live Market page) is responsible for falling back to
    the latest available HISTORICAL price/date in that case, clearly
    labeled as such.
    """
    provider = os.environ.get("MARKET_PROVIDER", "").strip().lower()
    if not provider:
        provider = "angelone" if os.environ.get("ANGELONE_API_KEY", "").strip() else "generic"

    if provider == "angelone":
        return _fetch_live_price_angelone(symbol)
    return _fetch_live_price_generic(symbol)


def _fetch_live_price_angelone(symbol: str) -> LivePriceResult:
    """
    Angel One SmartAPI provider. `symbol` is this project's internal
    company code (e.g. "RELI") - looked up via
    src/market/angelone_symbol_map.py to get the (exchange,
    tradingsymbol) pair Angel One actually needs.
    """
    mapping, reason = get_broker_symbol(symbol)
    if mapping is None:
        return _unavailable(reason)

    kind, value = mapping
    if kind == "NFO_FUT":
        # Dynamically-resolved front-month futures contract - see
        # angelone_symbol_map.py and angelone_client.get_ltp_future().
        quote = angelone_client.get_ltp_future(value)
    else:
        # kind is an Angel One exchange segment (e.g. "NSE"), value is the tradingsymbol.
        quote = angelone_client.get_ltp(kind, value)

    if not quote.available:
        return _unavailable(quote.reason or "Angel One live price unavailable.")

    return LivePriceResult(available=True, price=quote.price, timestamp=quote.timestamp,
                            open=quote.open, high=quote.high, low=quote.low,
                            previous_close=quote.previous_close)


def _fetch_live_price_generic(symbol: str) -> LivePriceResult:
    """
    Generic, provider-agnostic templated-URL client (MARKET_API_URL /
    MARKET_API_KEY / MARKET_API_PRICE_FIELD). See module docstring.

    BUG FIX (2026-09): this used to substitute `symbol` (this
    project's internal code, e.g. "RELI") straight into the URL
    template. That is NOT a real exchange ticker (see company_map.py's
    warning) - of this project's 50 codes, only 3 ("INFY", "ITC",
    "ONGC") happen to already equal their real NSE ticker, so a real
    provider would return data for those 3 and fail for the other 47.
    That is the most likely explanation for "some stocks show a live
    price and others don't" with the generic provider configured. Now
    `symbol`/`ticker` in the URL template both resolve to the real
    plain NSE ticker via src/market/nse_symbol_map.py (itself derived
    from angelone_symbol_map.py's best-effort-verified mappings)
    before the request is made - and, same as the Angel One path, a
    code that is unknown or deliberately unmapped (unverified
    underlying) honestly reports unavailable rather than sending a
    code that can never resolve to real data.
    """
    api_key = os.environ.get("MARKET_API_KEY", "").strip()
    url_template = os.environ.get("MARKET_API_URL", "").strip()
    price_field = os.environ.get("MARKET_API_PRICE_FIELD", "price").strip() or "price"
    symbol_suffix = os.environ.get("MARKET_API_SYMBOL_SUFFIX", "").strip()

    if not url_template:
        return _unavailable("Live market data API is not configured for this deployment.")

    ticker, reason = get_ticker(symbol)
    if ticker is None:
        return _unavailable(reason)
    ticker = ticker + symbol_suffix

    try:
        url = url_template.format(symbol=ticker, ticker=ticker, api_key=api_key)
    except (KeyError, IndexError):
        return _unavailable("Live market data API URL template is misconfigured.")

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return _unavailable("Connection/API unavailable.")

    raw_price = _resolve_field(payload, price_field)
    try:
        price = float(raw_price)
    except (TypeError, ValueError):
        return _unavailable("Live market data API response did not contain a usable price.")

    return LivePriceResult(available=True, price=price, timestamp=datetime.now(IST), reason=None)