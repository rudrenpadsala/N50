"""
src/market/nse_symbol_map.py

Maps this project's internal company codes (e.g. "RELI", "APLH" - see
src/data/company_map.py) to a PLAIN, exchange-suffix-free NSE ticker
(e.g. "RELIANCE", "APOLLOHOSP"), for use by ANY generic/third-party
market-data provider (see _fetch_live_price_generic in
src/market/live_data.py).

WHY THIS EXISTS
-----------------
This project's internal codes are investing.com export codes, not
real exchange ticker symbols (see company_map.py's own warning). Of
the 50 codes this project knows about, only 3 ("INFY", "ITC", "ONGC")
happen to already equal their real NSE ticker - the other 47 do not
(e.g. "RELI" is not "RELIANCE", "APLH" is not "APOLLOHOSP", "SUN" is
not "SUNPHARMA"). A generic provider that receives the raw internal
code will therefore return no data (or the wrong company's data, if a
provider is loose about matching) for most codes and silently
"work" only for the handful that coincidentally match - this is the
most likely explanation for "some stocks show a live price and others
don't" when the generic provider is configured. This module fixes
that by resolving every code to its real ticker before it ever
reaches a provider's URL.

SINGLE SOURCE OF TRUTH: rather than maintaining a second, possibly-
diverging translation table, this DERIVES the plain ticker from
src/market/angelone_symbol_map.py's already best-effort-verified
mappings (stripping Angel One's NSE cash-segment "-EQ" suffix for
equities; using the futures underlying's name directly for the
"c1_NS" futures-series codes, since Angel One's scrip-master "name"
field for a stock future is the same as that company's plain ticker).
That means the same "READ THIS BEFORE TRUSTING ANY LIVE PRICE" caveats
in angelone_symbol_map.py apply here too, and the same two
deliberately-unmapped codes (INGL, MAXE - unverified underlying
company) are deliberately unmapped here as well, for the same reason:
guessing wrong is worse than honestly reporting "unavailable".

Most generic providers also expect an exchange suffix (e.g. Yahoo-
style "RELIANCE.NS"). Rather than hardcoding one convention, callers
can append their provider's convention via the optional
MARKET_API_SYMBOL_SUFFIX environment variable (see live_data.py and
.env.example) - this module only resolves the bare ticker.
"""

from typing import Optional, Tuple

from src.market.angelone_symbol_map import SYMBOL_MAP


def get_ticker(code: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (ticker, None) - a plain NSE ticker with no exchange
    suffix, e.g. "RELIANCE" - or (None, reason) if `code` is unknown
    or deliberately unmapped. Never guesses.
    """
    if code not in SYMBOL_MAP:
        return None, f"'{code}' has no NSE ticker mapping configured (unknown internal code)."

    mapping = SYMBOL_MAP[code]
    if mapping is None:
        return None, (f"'{code}' is deliberately left unmapped (futures contract or unverified "
                       f"underlying - see angelone_symbol_map.py's header).")

    kind, value = mapping
    if kind == "NFO_FUT":
        # `value` is already Angel One's scrip-master "name" field for the
        # underlying, e.g. "ADANIENT" - the same string used as that
        # company's plain ticker.
        return value, None

    # `value` is an Angel One NSE cash-segment tradingsymbol, e.g.
    # "RELIANCE-EQ" - strip the "-EQ" suffix to get the plain ticker.
    tradingsymbol = value
    ticker = tradingsymbol[:-3] if tradingsymbol.endswith("-EQ") else tradingsymbol
    return ticker, None
