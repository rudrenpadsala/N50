"""
src/market/angelone_client.py

Minimal Angel One SmartAPI REST client for read-only LTP (last traded
price) quotes. This is the concrete provider behind
`src/market/live_data.py` when Angel One credentials are configured.

WHY A HAND-ROLLED CLIENT INSTEAD OF THE `smartapi-python` SDK
---------------------------------------------------------------
The official SDK pulls in extra dependencies (websocket-client,
logzero, etc.) this project doesn't otherwise need for a single
read-only price lookup. Angel One's REST endpoints are simple, stable,
and documented, so we call them directly with `requests`. If you later
need order placement, historical candles, or the websocket feed,
switching to `smartapi-python` (https://github.com/angel-one/smartapi-python)
is a reasonable upgrade.

AUTHENTICATION MODEL (per Angel One's SmartAPI docs, current as of
this writing - Angel One can change this without notice, so if login
starts failing, check https://smartapi.angelone.in/docs/User first)
---------------------------------------------------------------
Angel One retired plain-password API login; you authenticate with:
    1. clientcode   - your Angel One client ID (e.g. "A123456")
    2. password     - your trading PIN (NOT your web-login password)
    3. totp         - a 6-digit code from an authenticator app, generated
                       from a TOTP *secret* you obtain once by scanning
                       the QR code at https://smartapi.angelbroking.com/enable-totp

This module never stores the 6-digit code - you give it the TOTP
*secret* (ANGELONE_TOTP_SECRET) and it generates a fresh code on every
login using `pyotp`, exactly like your authenticator app does.

A successful login returns a JWT ("jwtToken") used as a Bearer token
on all subsequent calls, a refreshToken, and a feedToken (for the
websocket feed, unused here). Angel One sessions are valid until
~midnight IST or until explicitly logged out; this module caches the
session in-process for a few hours and re-logs-in transparently on
expiry or on an auth-error response code, so a Streamlit app calling
`get_ltp()` repeatedly does not need to worry about the login flow at
all.

NEVER FABRICATES A PRICE: every failure path (missing credentials, bad
TOTP secret, network failure, malformed response, unknown symbol,
rate limiting) returns an `AngelOneQuote(available=False, reason=...)`
- it never raises out to the caller for expected failure modes, and
never invents a plausible-looking number.
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import requests

try:
    import pyotp
except ImportError:  # pragma: no cover - exercised only if pyotp isn't installed
    pyotp = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://apiconnect.angelone.in"
LOGIN_PATH = "/rest/auth/angelbroking/user/v1/loginByPassword"
LTP_PATH = "/rest/secure/angelbroking/order/v1/getLtpData"
HISTORICAL_PATH = "/rest/secure/angelbroking/historical/v1/getCandleData"

# Public instrument master (symbol -> token). ~30 MB; cached to disk and
# refreshed at most once a day, since tokens rarely change intraday.
SCRIP_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

REQUEST_TIMEOUT_SECONDS = 10
# The historical-candle endpoint returns much more data per call (up to
# ~120 days of OHLCV) than a single LTP lookup, and has been observed to
# be slower/flakier under load - a longer timeout cuts down on "Could
# not reach..." failures that were really just a slow-but-working
# response getting cut off too early.
HISTORICAL_REQUEST_TIMEOUT_SECONDS = 20
SCRIP_MASTER_DOWNLOAD_TIMEOUT_SECONDS = 60

# BUG FIX (2026-09): a single transient network blip (timeout, reset
# connection, or Angel One returning a non-JSON body under load - all
# observed in practice, especially on the historical-candle endpoint)
# used to fail that one request outright with no retry, while the very
# same stock would often succeed a moment later on manual refresh. This
# looks exactly like "some stocks show a live price/decision and others
# don't" even though every stock is equally capable of working - it's
# just a matter of which request happened to land during a blip. One
# short, automatic retry closes most of that gap without masking a
# genuinely persistent failure (credentials, mapping, etc. still fail
# immediately as before).
NETWORK_RETRY_ATTEMPTS = 2
NETWORK_RETRY_BACKOFF_SECONDS = 1.5

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIP_MASTER_CACHE_PATH = _PROJECT_ROOT / "data" / "cache" / "angelone_scrip_master.json"
SCRIP_MASTER_MAX_AGE_HOURS = 20

# Angel One error codes that mean "your token is bad/expired" - worth one
# transparent re-login retry rather than surfacing as a hard failure.
AUTH_ERROR_CODES = {"AG8001", "AG8002", "AG8003"}


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

@dataclass
class AngelOneCredentials:
    api_key: str
    client_code: str
    password: str       # trading PIN, not the web-login password
    totp_secret: str    # base32 TOTP secret from the QR-code enrollment step


def _load_credentials_from_env() -> Optional[AngelOneCredentials]:
    api_key = os.environ.get("ANGELONE_API_KEY", "").strip()
    client_code = os.environ.get("ANGELONE_CLIENT_CODE", "").strip()
    password = os.environ.get("ANGELONE_PASSWORD", "").strip()
    totp_secret = os.environ.get("ANGELONE_TOTP_SECRET", "").strip()
    if not (api_key and client_code and password and totp_secret):
        return None
    return AngelOneCredentials(api_key=api_key, client_code=client_code,
                                password=password, totp_secret=totp_secret)


def _client_headers(api_key: str, jwt_token: Optional[str] = None) -> dict:
    """
    Angel One requires these X- headers on every call (login and
    secure). Local/public IP and MAC address are best-effort and NOT
    verified by Angel One for a server-side deployment - real values
    can be supplied via env vars if you ever see requests rejected for
    "invalid header" reasons, but placeholders work for LTP reads in
    practice.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-UserType": "USER",
        "X-SourceID": "WEB",
        "X-ClientLocalIP": os.environ.get("ANGELONE_CLIENT_LOCAL_IP", "127.0.0.1"),
        "X-ClientPublicIP": os.environ.get("ANGELONE_CLIENT_PUBLIC_IP", "127.0.0.1"),
        "X-MACAddress": os.environ.get("ANGELONE_MAC_ADDRESS", "00:00:00:00:00:00"),
        "X-PrivateKey": api_key,
    }
    if jwt_token:
        headers["Authorization"] = f"Bearer {jwt_token}"
    return headers


def _post_json_with_retry(url: str, payload: dict, headers: dict, timeout: int,
                           endpoint_label: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    POSTs `payload` and parses the JSON response, retrying once after a
    short backoff on a transient failure: a timeout, a connection
    error, or a response body that isn't valid JSON (Angel One has been
    observed to return a non-JSON body - e.g. a plain-text gateway
    error - when it's under load or rate-limiting a client, especially
    on the historical-candle endpoint). Returns (body, None) or
    (None, reason) - the reason names which kind of failure it was, so
    a persistent problem (vs. a one-off blip) is easy to tell apart in
    the UI. This is deliberately separate from the auth-retry logic in
    get_ltp/get_historical_candles below, which retries on a *valid*
    response that says the session itself is invalid - this retries on
    not getting a usable response at all.
    """
    last_reason = "unknown error"
    for attempt in range(NETWORK_RETRY_ATTEMPTS):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.Timeout:
            last_reason = f"timed out after {timeout}s"
        except requests.RequestException as exc:
            last_reason = f"connection error ({exc.__class__.__name__})"
        else:
            try:
                return response.json(), None
            except ValueError:
                # BUG FIX (2026-09): show WHAT Angel One actually sent back
                # instead of a generic "non-JSON" message. A non-JSON body
                # is almost always one of: (a) a genuine rate-limit page,
                # (b) an auth/subscription gateway page (e.g. this API
                # feature isn't enabled for this app in the Angel One
                # developer portal), or (c) an IP/compliance block page.
                # The HTTP status code and a short text snippet reliably
                # distinguish these without needing another round trip.
                snippet = (response.text or "").strip().replace("\n", " ")[:160]
                last_reason = (f"HTTP {response.status_code}, returned a non-JSON response"
                                f"{f': \"{snippet}\"' if snippet else ' (empty body)'}")
                if response.status_code == 403 and "exceeding access rate" in snippet.lower():
                    # BUG FIX (2026-09): a "403 - exceeding access rate"
                    # response is Angel One's own rate limiter, not a blip -
                    # retrying 1.5s later is certain to hit the exact same
                    # block, so it only adds delay without any chance of
                    # succeeding. Stop immediately instead of burning the
                    # retry, and say plainly that this is a rate limit
                    # (spread requests out / wait a bit) rather than
                    # something to keep hammering.
                    return None, (f"Could not reach Angel One's {endpoint_label} endpoint "
                                   f"(rate-limited by Angel One - \"{snippet}\". This is Angel One's "
                                   f"own request-rate limit, not a bug here; wait a minute before "
                                   f"checking more stocks, or check your API key's rate-limit tier "
                                   f"in the Angel One developer portal).")
        if attempt < NETWORK_RETRY_ATTEMPTS - 1:
            time.sleep(NETWORK_RETRY_BACKOFF_SECONDS)
    return None, f"Could not reach Angel One's {endpoint_label} endpoint ({last_reason})."


# ---------------------------------------------------------------------------
# Session (login) handling
# ---------------------------------------------------------------------------

@dataclass
class AngelOneSession:
    jwt_token: str
    refresh_token: Optional[str]
    feed_token: Optional[str]
    fetched_at: datetime

    def is_stale(self, max_age_minutes: int = 6 * 60) -> bool:
        return datetime.now() - self.fetched_at > timedelta(minutes=max_age_minutes)


_session_cache: Optional[AngelOneSession] = None


def _login(creds: AngelOneCredentials) -> Tuple[Optional[AngelOneSession], Optional[str]]:
    if pyotp is None:
        return None, "The 'pyotp' package is required for Angel One login but is not installed (pip install pyotp)."

    try:
        totp_code = pyotp.TOTP(creds.totp_secret).now()
    except Exception:
        return None, "ANGELONE_TOTP_SECRET is not a valid base32 TOTP secret."

    payload = {
        "clientcode": creds.client_code,
        "password": creds.password,
        "totp": totp_code,
    }
    try:
        response = requests.post(
            BASE_URL + LOGIN_PATH,
            json=payload,
            headers=_client_headers(creds.api_key),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        body = response.json()
    except (requests.RequestException, ValueError):
        return None, "Could not reach Angel One's login endpoint."

    if not isinstance(body, dict) or not body.get("status"):
        message = body.get("message", "unknown error") if isinstance(body, dict) else "unexpected response"
        return None, f"Angel One login failed: {message}"

    data = body.get("data") or {}
    jwt_token = data.get("jwtToken")
    if not jwt_token:
        return None, "Angel One login response did not contain a session token."

    session = AngelOneSession(
        jwt_token=jwt_token,
        refresh_token=data.get("refreshToken"),
        feed_token=data.get("feedToken"),
        fetched_at=datetime.now(),
    )
    return session, None


def _get_session(force: bool = False) -> Tuple[Optional[AngelOneSession], Optional[str]]:
    global _session_cache

    creds = _load_credentials_from_env()
    if creds is None:
        return None, ("Angel One credentials are not configured (need ANGELONE_API_KEY, "
                       "ANGELONE_CLIENT_CODE, ANGELONE_PASSWORD, ANGELONE_TOTP_SECRET).")

    if not force and _session_cache is not None and not _session_cache.is_stale():
        return _session_cache, None

    session, err = _login(creds)
    if session is None:
        return None, err
    _session_cache = session
    return session, None


def reset_session_cache() -> None:
    """Force the next call to log in again. Mostly useful for tests."""
    global _session_cache
    _session_cache = None


# ---------------------------------------------------------------------------
# Scrip (instrument) master - symbol -> token lookup
# ---------------------------------------------------------------------------

_scrip_master_memory_cache: Optional[list] = None


def _load_scrip_master() -> Tuple[Optional[list], Optional[str]]:
    """
    Returns the full instrument list, downloading and disk-caching it
    if needed. Falls back to a stale on-disk cache if a fresh download
    fails, rather than failing outright - a day-old token map is still
    almost certainly correct.
    """
    global _scrip_master_memory_cache
    if _scrip_master_memory_cache is not None:
        return _scrip_master_memory_cache, None

    try:
        SCRIP_MASTER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # fall through - we can still try an in-memory-only fetch

    if SCRIP_MASTER_CACHE_PATH.exists():
        try:
            age_hours = (time.time() - SCRIP_MASTER_CACHE_PATH.stat().st_mtime) / 3600
        except OSError:
            age_hours = None
        if age_hours is not None and age_hours < SCRIP_MASTER_MAX_AGE_HOURS:
            try:
                with open(SCRIP_MASTER_CACHE_PATH, "r") as f:
                    data = json.load(f)
                _scrip_master_memory_cache = data
                return data, None
            except (ValueError, OSError):
                pass  # cache file is corrupt/unreadable - fall through to re-download

    try:
        response = requests.get(SCRIP_MASTER_URL, timeout=SCRIP_MASTER_DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        # Download failed - use a stale cache if we have one instead of giving up.
        if SCRIP_MASTER_CACHE_PATH.exists():
            try:
                with open(SCRIP_MASTER_CACHE_PATH, "r") as f:
                    data = json.load(f)
                _scrip_master_memory_cache = data
                return data, None
            except (ValueError, OSError):
                pass
        return None, f"Could not download Angel One's instrument list ({exc})."

    try:
        with open(SCRIP_MASTER_CACHE_PATH, "w") as f:
            json.dump(data, f)
    except OSError:
        pass  # non-fatal - we still have it in memory for this run

    _scrip_master_memory_cache = data
    return data, None


def lookup_symbol_token(exchange: str, tradingsymbol: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Look up the numeric `symboltoken` Angel One requires alongside
    exchange + tradingsymbol for the LTP call. Returns (token, None) or
    (None, reason).

    Matching is case/whitespace-normalized (BUG FIX 2026-09: a strict
    `==` comparison meant a stray space or a casing difference between
    angelone_symbol_map.py's best-effort tradingsymbol and Angel One's
    own scrip-master entry - both plausible for a mapping that hasn't
    been individually verified per that file's own warning - silently
    failed this specific stock's live price while every exactly-correct
    mapping kept working, which looks exactly like "some stocks show a
    live price and others don't"). This never changes WHICH instrument
    matches, only how forgiving the text comparison is.
    """
    scrips, err = _load_scrip_master()
    if scrips is None:
        return None, err

    exchange_norm = exchange.strip().upper()
    tradingsymbol_norm = tradingsymbol.strip().upper()
    for row in scrips:
        row_exchange = str(row.get("exch_seg") or "").strip().upper()
        row_symbol = str(row.get("symbol") or "").strip().upper()
        if row_exchange == exchange_norm and row_symbol == tradingsymbol_norm:
            return row.get("token"), None

    return None, f"'{tradingsymbol}' on exchange '{exchange}' was not found in Angel One's instrument list."


# ---------------------------------------------------------------------------
# Futures - resolving the current (front-month) contract dynamically
# ---------------------------------------------------------------------------
#
# Stock-futures contracts expire monthly, so hardcoding a tradingsymbol
# like "ADANIENT25SEPFUT" in angelone_symbol_map.py would go stale
# every month. Instead, angelone_symbol_map.py stores the *underlying*
# name (Angel One's scrip-master "name" field, e.g. "ADANIENT" - NOT
# the "-EQ" symbol), and this function finds whichever NFO FUTSTK
# contract for that name currently has the nearest (soonest, but not
# yet expired) expiry - i.e. the front-month contract - every time
# it's called. This mirrors how a "continuous futures" price series
# (the kind these c1_NS-suffixed codes represent) is usually built:
# it tracks the front-month contract and rolls to the next one at
# expiry.

_EXPIRY_FORMATS = ("%d%b%Y", "%d%b%y", "%Y-%m-%d")


def _parse_expiry(raw) -> Optional[date]:
    if not raw:
        return None
    text = str(raw).strip()
    for fmt in _EXPIRY_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def resolve_front_month_future(underlying_name: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Finds the nearest-expiry, not-yet-expired NFO FUTSTK contract for
    `underlying_name` (e.g. "RELIANCE", "ADANIENT" - Angel One's
    scrip-master "name" field, not a "-EQ" tradingsymbol). Returns
    ({"tradingsymbol": ..., "token": ..., "expiry": date(...)}, None)
    or (None, reason).
    """
    scrips, err = _load_scrip_master()
    if scrips is None:
        return None, err

    today = datetime.now().date()
    underlying_norm = underlying_name.strip().upper()
    candidates = []
    for row in scrips:
        if row.get("exch_seg") != "NFO" or row.get("instrumenttype") != "FUTSTK":
            continue
        # Case/whitespace-normalized match - see lookup_symbol_token's
        # docstring above for why a strict `==` here would silently
        # break individual futures codes while others kept working.
        if str(row.get("name") or "").strip().upper() != underlying_norm:
            continue
        expiry_date = _parse_expiry(row.get("expiry"))
        if expiry_date is None or expiry_date < today:
            continue
        candidates.append((expiry_date, row))

    if not candidates:
        return None, f"No live NFO futures contract found for underlying '{underlying_name}'."

    candidates.sort(key=lambda pair: pair[0])
    nearest_expiry, nearest_row = candidates[0]
    return {
        "tradingsymbol": nearest_row.get("symbol"),
        "token": nearest_row.get("token"),
        "expiry": nearest_expiry,
    }, None


def get_ltp_future(underlying_name: str) -> "AngelOneQuote":
    """
    Fetch LTP for the front-month NFO stock future of `underlying_name`.
    Resolves the current live contract automatically on every call, so
    this does not go stale when a contract expires and rolls over.
    """
    creds = _load_credentials_from_env()
    if creds is None:
        return AngelOneQuote(available=False, reason="Angel One credentials are not configured.")

    contract, err = resolve_front_month_future(underlying_name)
    if contract is None:
        return AngelOneQuote(available=False, reason=err)

    return get_ltp("NFO", contract["tradingsymbol"], symboltoken=contract["token"])


# ---------------------------------------------------------------------------
# LTP (last traded price)
# ---------------------------------------------------------------------------

@dataclass
class AngelOneQuote:
    available: bool
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    reason: Optional[str] = None
    # BUG FIX (2026-09): Angel One's getLtpData response already includes
    # open/high/low/close (previous day's close) alongside ltp, in the
    # SAME call - see get_ltp() below. Surfacing them here means "today's
    # change" and "previous close" no longer depend on the separate
    # historical-candle endpoint, which has a long-standing, currently
    # unresolved bug on Angel One's own platform where it falsely
    # rejects requests as rate-limited even far below their documented
    # limits (confirmed via multiple independent reports on Angel One's
    # own SmartAPI developer forum, including their own support staff
    # declining to help debug it). The LTP endpoint does not share this
    # bug and was already working reliably.
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    previous_close: Optional[float] = None


def get_ltp(exchange: str, tradingsymbol: str, symboltoken: Optional[str] = None) -> AngelOneQuote:
    """
    Fetch the last traded price for one instrument. `symboltoken` is
    looked up automatically from the scrip master if not supplied.
    Never raises for expected failure modes - always returns an
    AngelOneQuote with available=False and a human-readable reason.
    """
    creds = _load_credentials_from_env()
    if creds is None:
        return AngelOneQuote(available=False,
                              reason="Angel One credentials are not configured.")

    session, err = _get_session()
    if session is None:
        return AngelOneQuote(available=False, reason=err)

    if not symboltoken:
        symboltoken, tok_err = lookup_symbol_token(exchange, tradingsymbol)
        if not symboltoken:
            return AngelOneQuote(available=False, reason=tok_err)

    payload = {"exchange": exchange, "tradingsymbol": tradingsymbol, "symboltoken": str(symboltoken)}
    url = BASE_URL + LTP_PATH

    body, reason = _post_json_with_retry(
        url, payload, _client_headers(creds.api_key, session.jwt_token), REQUEST_TIMEOUT_SECONDS, "quote")
    if body is None:
        return AngelOneQuote(available=False, reason=reason)

    # Session expired/invalidated server-side - retry once with a fresh login
    # rather than surfacing a confusing auth error to the end user.
    if isinstance(body, dict) and not body.get("status") and body.get("errorcode") in AUTH_ERROR_CODES:
        session, err = _get_session(force=True)
        if session is None:
            return AngelOneQuote(available=False, reason=err)
        body, reason = _post_json_with_retry(
            url, payload, _client_headers(creds.api_key, session.jwt_token), REQUEST_TIMEOUT_SECONDS, "quote")
        if body is None:
            return AngelOneQuote(available=False, reason=reason)

    if not isinstance(body, dict) or not body.get("status"):
        message = body.get("message", "unknown error") if isinstance(body, dict) else "unexpected response"
        return AngelOneQuote(available=False, reason=f"Angel One error: {message}")

    data = body.get("data") or {}
    try:
        price = float(data.get("ltp"))
    except (TypeError, ValueError):
        return AngelOneQuote(available=False, reason="Angel One response did not contain a usable LTP.")

    def _optional_float(key: str) -> Optional[float]:
        try:
            return float(data.get(key))
        except (TypeError, ValueError):
            return None

    return AngelOneQuote(available=True, price=price, timestamp=datetime.now(),
                          open=_optional_float("open"), high=_optional_float("high"),
                          low=_optional_float("low"), previous_close=_optional_float("close"))


# ---------------------------------------------------------------------------
# Historical daily candles (used to compute "live" technical indicators
# for today, rather than being frozen at the static training dataset's
# last date - see src/decision/live_features.py)
# ---------------------------------------------------------------------------

def get_historical_candles(
    exchange: str, symboltoken: str, from_date: datetime, to_date: datetime,
) -> Tuple[Optional[list], Optional[str]]:
    """
    Fetch daily OHLCV candles for one instrument between `from_date` and
    `to_date` (inclusive, both naive datetimes interpreted as IST - Angel
    One does not accept timezone-aware timestamps here). Returns
    (candles, None) where each candle is
    [timestamp_str, open, high, low, close, volume], oldest first - or
    (None, reason) on any failure. Never fabricates data.
    """
    creds = _load_credentials_from_env()
    if creds is None:
        return None, "Angel One credentials are not configured."

    session, err = _get_session()
    if session is None:
        return None, err

    payload = {
        "exchange": exchange,
        "symboltoken": str(symboltoken),
        "interval": "ONE_DAY",
        "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
        "todate": to_date.strftime("%Y-%m-%d %H:%M"),
    }
    url = BASE_URL + HISTORICAL_PATH

    body, reason = _post_json_with_retry(
        url, payload, _client_headers(creds.api_key, session.jwt_token),
        HISTORICAL_REQUEST_TIMEOUT_SECONDS, "historical-data")
    if body is None:
        return None, reason

    if isinstance(body, dict) and not body.get("status") and body.get("errorcode") in AUTH_ERROR_CODES:
        session, err = _get_session(force=True)
        if session is None:
            return None, err
        body, reason = _post_json_with_retry(
            url, payload, _client_headers(creds.api_key, session.jwt_token),
            HISTORICAL_REQUEST_TIMEOUT_SECONDS, "historical-data")
        if body is None:
            return None, reason

    if not isinstance(body, dict) or not body.get("status"):
        message = body.get("message", "unknown error") if isinstance(body, dict) else "unexpected response"
        return None, f"Angel One historical-data error: {message}"

    candles = body.get("data")
    if not candles:
        return None, "Angel One returned no candle data for this date range."

    return candles, None

