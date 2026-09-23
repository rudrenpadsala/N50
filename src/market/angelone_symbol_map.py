"""
src/market/angelone_symbol_map.py

Maps this project's internal company codes (see src/data/company_map.py,
e.g. "RELI", "INFY") to the (exchange, tradingsymbol) pair Angel One's
SmartAPI needs for a quote (e.g. ("NSE", "RELIANCE-EQ")).

READ THIS BEFORE TRUSTING ANY LIVE PRICE FOR A REAL TRADING DECISION
---------------------------------------------------------------------
1. This project's codes are investing.com export codes (see
   company_map.py's own warning), not exchange ticker symbols. The
   mappings below are a best-effort match to the NSE EQUITY (cash)
   symbol for that company, based on the company name company_map.py
   already resolved - they have NOT been individually verified against
   Angel One's live scrip master by a human.
2. Several of this project's codes are FUTURES-CONTRACT series
   (suffix "c1_NS" - see company_map.py), not the underlying equity.
   A futures price is NOT the same instrument as the spot/EQ price and
   can differ meaningfully, especially near expiry or high volatility.
   These are mapped below to ("NFO_FUT", <underlying name>) - a marker
   that tells live_data.py to resolve the current front-month NFO
   futures contract for that underlying dynamically, via
   angelone_client.resolve_front_month_future(), on every call. This
   avoids hardcoding a contract symbol (e.g. "ADANIENT25SEPFUT") that
   would go stale the moment that contract expires and the next
   month's contract becomes current.
3. "INGL" and "MAXE" are flagged UNVERIFIED in company_map.py itself
   (the underlying company could not be identified with confidence) -
   left unmapped here for the same reason.
4. BEFORE GOING LIVE: verify every mapping you intend to use against
   Angel One's own instrument list.
   For equities:
       from src.market.angelone_client import lookup_symbol_token
       lookup_symbol_token("NSE", "RELIANCE-EQ")
   should return a token; if it returns an error, the tradingsymbol
   below is wrong for that exchange segment and must be corrected
   before this feeds any BUY/SELL decision.
   For futures:
       from src.market.angelone_client import resolve_front_month_future
       resolve_front_month_future("ADANIENT")
   should return a contract dict; if it returns an error, the
   underlying name below doesn't match Angel One's "name" field and
   must be corrected.

Format: CODE -> (exchange_segment, tradingsymbol), ("NFO_FUT", underlying_name)
for a dynamically-resolved futures contract, or None if deliberately
unmapped. "exchange_segment" matches Angel One's "exch_seg" field:
"NSE" for cash-market equities.
"""

from typing import Optional, Tuple

SYMBOL_MAP: dict = {
    # --- NSE cash-market equities (best-effort match - verify before live use) ---
    "APLH": ("NSE", "APOLLOHOSP-EQ"),
    "ASPN": ("NSE", "ASIANPAINT-EQ"),
    "AXBK": ("NSE", "AXISBANK-EQ"),
    "BAJE": ("NSE", "BAJAJFINSV-EQ"),
    "BJFN": ("NSE", "BAJFINANCE-EQ"),
    "BJFS": ("NSE", "BAJAJFINSV-EQ"),  # company_map.py notes this as an alt code for the same company as BAJE
    "BRTI": ("NSE", "BHARTIARTL-EQ"),
    "CIPL": ("NSE", "CIPLA-EQ"),
    "COAL": ("NSE", "COALINDIA-EQ"),
    "EICH": ("NSE", "EICHERMOT-EQ"),
    "ETEA": ("NSE", "ETERNAL-EQ"),
    "GRAS": ("NSE", "GRASIM-EQ"),
    "HALC": ("NSE", "HINDALCO-EQ"),
    "HCLT": ("NSE", "HCLTECH-EQ"),
    "HLL": ("NSE", "HINDUNILVR-EQ"),
    "ICBK": ("NSE", "ICICIBANK-EQ"),
    "INFY": ("NSE", "INFY-EQ"),
    "ITC": ("NSE", "ITC-EQ"),
    "JIOF": ("NSE", "JIOFIN-EQ"),
    "JSTL": ("NSE", "JSWSTEEL-EQ"),
    "LART": ("NSE", "LT-EQ"),
    "MAHM": ("NSE", "M&M-EQ"),
    "MRTI": ("NSE", "MARUTI-EQ"),
    "NEST": ("NSE", "NESTLEIND-EQ"),
    "ONGC": ("NSE", "ONGC-EQ"),
    "REDY": ("NSE", "DRREDDY-EQ"),
    "RELI": ("NSE", "RELIANCE-EQ"),
    "SHMF": ("NSE", "SHRIRAMFIN-EQ"),
    "SUN": ("NSE", "SUNPHARMA-EQ"),
    "TACN": ("NSE", "TATACONSUM-EQ"),
    # BUG FIX (2026-09): Tata Motors Ltd was demerged on 1 Oct 2025 and
    # renamed Tata Motors Passenger Vehicles Ltd on 24 Oct 2025 - its NSE
    # symbol changed from TATAMOTORS to TMPV (the commercial-vehicle
    # business was separately spun off in Nov 2025 as a NEW listing,
    # eventually renamed "Tata Motors Ltd" again under symbol TMCV - a
    # DIFFERENT company from this project's "TAMO" code). "TATAMOTORS"
    # no longer exists on NSE, which is why this code's live price
    # always failed with "not found in Angel One's instrument list".
    # Confirmed against this project's own historical CSV: its price
    # series drops from ~434 to ~400 exactly on the 14-Oct-2025 record
    # date (matching the reported ~660->400 post-demerger adjustment)
    # and continues under that level through 2026 - i.e. this code has
    # always tracked the passenger-vehicle entity, now called TMPV.
    "TAMO": ("NSE", "TMPV-EQ"),
    "TEML": ("NSE", "TECHM-EQ"),
    "TISC": ("NSE", "TATASTEEL-EQ"),
    "TITN": ("NSE", "TITAN-EQ"),
    "TREN": ("NSE", "TRENT-EQ"),
    "ULTC": ("NSE", "ULTRACEMCO-EQ"),
    "WIPR": ("NSE", "WIPRO-EQ"),

    # --- Futures-series codes (company_map.py "c1_NS" suffix): resolved
    #     dynamically to the current front-month NFO contract - see
    #     resolve_front_month_future() in angelone_client.py. The second
    #     element is Angel One's scrip-master "name" field for the
    #     underlying (verified against NSE's official symbol list). ---
    "ADELc1_NS": ("NFO_FUT", "ADANIENT"),      # Adani Enterprises
    "APSEc1_NS": ("NFO_FUT", "ADANIPORTS"),    # Adani Ports & SEZ
    "BAJAc1_NS": ("NFO_FUT", "BAJAJ-AUTO"),    # Bajaj Auto
    "HDBKc1_NS": ("NFO_FUT", "HDFCBANK"),      # HDFC Bank
    "HDFLc1_NS": ("NFO_FUT", "HDFCLIFE"),      # HDFC Life Insurance
    "KTKMc1_NS": ("NFO_FUT", "KOTAKBANK"),     # Kotak Mahindra Bank
    "NTPCc1_NS": ("NFO_FUT", "NTPC"),          # NTPC
    "PGRDc1_NS": ("NFO_FUT", "POWERGRID"),     # Power Grid Corporation
    "SBIc1_NS": ("NFO_FUT", "SBIN"),           # State Bank of India
    "TAMOc1_NS": ("NFO_FUT", "TMPV"),          # Tata Motors -> renamed TMPV, see TAMO above
    "TCSc1_NS": ("NFO_FUT", "TCS"),            # Tata Consultancy Services

    # --- Unverified underlying company (flagged in company_map.py itself) ---
    "INGL": None,
    "MAXE": None,
}


def get_broker_symbol(code: str) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
    """
    Returns ((exchange, tradingsymbol), None) if this internal code has
    a configured Angel One mapping, or (None, reason) otherwise -
    covering both "we've never heard of this code" and "we know this
    code but deliberately left it unmapped" (futures/unverified).
    """
    if code not in SYMBOL_MAP:
        return None, f"'{code}' has no Angel One symbol mapping configured (unknown internal code)."

    mapping = SYMBOL_MAP[code]
    if mapping is None:
        return None, (f"'{code}' is deliberately left unmapped in angelone_symbol_map.py "
                       f"(futures contract or unverified underlying - see that file's header).")

    return mapping, None
