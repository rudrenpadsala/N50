"""
src/news/company_sector_map.py

Company -> sector metadata for the 3-tier News & Risk Layer (COMPANY /
SECTOR / MACRO - see src/news/news_analyzer.py). This does NOT create a
new company list: every key here is exactly one of the Symbols already
defined in src/data/company_map.CODE_MAP (the project's single source
of truth for which 47 companies exist). This module only ADDS metadata
- ticker, sector, industry, subsector, search aliases - on top of that
existing list.

WHY A SEPARATE FILE (not folded into company_map.py): company_map.py's
job is "raw source filename -> Symbol + best-effort display name" for
the DATA pipeline (Sprint 1) and is deliberately minimal. This file's
job is "Symbol -> everything the NEWS pipeline needs to know" (a real
ticker for the news search, a sector for the sector-tier search, macro
terms relevant to THAT sector) - a different concern, added later, with
no reason to touch the original.

SECTOR TAXONOMY: a fixed, small set of sector buckets (GICS-like, not
GICS itself) is defined once in SECTOR_SEARCH_TERMS below, each with:
  - sector_terms: what to search for SECTOR-tier news (per project
    spec: broad enough to catch sector news that never mentions the
    company by name, narrow enough not to pull in unrelated business
    news - e.g. "healthcare sector India", not "business").
  - macro_terms: which MACRO-tier searches actually matter for this
    sector (per spec: "Do not treat every RBI or Nifty article as
    relevant" - a rate-sensitive sector like Financial Services cares
    about RBI rate moves far more than, say, FMCG does).
A COMMON_MACRO_TERMS list (broad India/global market-moving events) is
searched for every company regardless of sector, per project spec's
level-3 examples (RBI, major government policy, tax changes, inflation).

CONFIDENCE: two Symbols (INGL, MAXE) are already flagged "UNVERIFIED"
in company_map.py itself - their sector/ticker guesses here are marked
low_confidence=True so the news layer can (optionally) surface that
caveat rather than presenting a guess as fact.
"""

from src.data.company_map import CODE_MAP

# ---------------------------------------------------------------------------
# Sector taxonomy: sector -> (sector_terms, extra_macro_terms)
# ---------------------------------------------------------------------------
# extra_macro_terms are IN ADDITION to COMMON_MACRO_TERMS (searched for
# every company) - these are the macro topics that specifically move
# this sector more than the market as a whole.

SECTOR_SEARCH_TERMS = {
    "Financial Services": {
        "sector_terms": [
            "Indian banking sector", "NBFC sector India", "RBI banking regulation",
            "bank credit growth India", "Indian financial services industry",
        ],
        "extra_macro_terms": [
            "RBI monetary policy", "RBI repo rate", "India interest rates",
            "India credit growth", "banking regulation India",
        ],
    },
    "Information Technology": {
        "sector_terms": [
            "Indian IT sector", "India IT services industry", "IT sector hiring India",
            "India software exports", "global tech spending outlook",
        ],
        "extra_macro_terms": [
            "US IT spending", "H-1B visa policy", "US recession tech spending",
            "rupee dollar exchange rate",
        ],
    },
    "Healthcare": {
        "sector_terms": [
            "Indian healthcare sector", "India hospital industry", "India pharma sector",
            "India healthcare policy", "drug pricing India",
        ],
        "extra_macro_terms": [
            "India healthcare policy", "India drug price control", "US FDA India pharma",
            "India health insurance regulation",
        ],
    },
    "Consumer Staples": {
        "sector_terms": [
            "India FMCG sector", "India consumer goods demand", "rural demand India FMCG",
            "India retail inflation consumer goods",
        ],
        "extra_macro_terms": [
            "India inflation", "India rural demand", "GST rate FMCG India",
            "India consumer spending",
        ],
    },
    "Consumer Discretionary": {
        "sector_terms": [
            "India retail sector", "India consumer discretionary spending",
            "India jewellery and apparel demand", "India retail industry",
        ],
        "extra_macro_terms": [
            "India consumer confidence", "India festive season demand", "GST rate India retail",
        ],
    },
    "Automobile": {
        "sector_terms": [
            "Indian automobile sector", "India auto sales", "India EV policy",
            "auto sector India demand",
        ],
        "extra_macro_terms": [
            "India auto policy", "India fuel prices", "EV subsidy India",
            "India auto import tariff",
        ],
    },
    "Materials": {
        "sector_terms": [
            "India metals sector", "India steel industry", "India cement sector",
            "commodity prices India metals",
        ],
        "extra_macro_terms": [
            "global commodity prices", "China demand metals", "India infrastructure spending",
        ],
    },
    "Energy": {
        "sector_terms": [
            "India oil and gas sector", "India energy sector", "crude oil prices India",
            "India coal industry",
        ],
        "extra_macro_terms": [
            "crude oil prices", "OPEC production", "India energy policy", "India fuel subsidy",
        ],
    },
    "Utilities": {
        "sector_terms": [
            "India power sector", "India electricity demand", "India power generation industry",
        ],
        "extra_macro_terms": [
            "India power policy", "India renewable energy policy", "coal supply India power",
        ],
    },
    "Industrials": {
        "sector_terms": [
            "India infrastructure sector", "India capital goods industry",
            "India engineering and construction sector",
        ],
        "extra_macro_terms": [
            "India infrastructure spending", "India capex budget", "government infrastructure policy India",
        ],
    },
    "Telecommunications": {
        "sector_terms": [
            "India telecom sector", "India telecom tariffs", "5G rollout India",
        ],
        "extra_macro_terms": [
            "India telecom regulation", "TRAI tariff", "spectrum auction India",
        ],
    },
    "Diversified/Conglomerate": {
        "sector_terms": [
            "India conglomerate business group", "India diversified industrial group",
        ],
        "extra_macro_terms": [
            "India corporate regulation", "India business conglomerate policy",
        ],
    },
    "Consumer Services": {
        "sector_terms": [
            "India internet sector", "India food delivery industry", "India e-commerce sector",
        ],
        "extra_macro_terms": [
            "India digital economy policy", "India e-commerce regulation",
        ],
    },
}

# Searched for EVERY company regardless of sector (project spec's
# level-3 examples verbatim): RBI, major government policy, tax
# changes, inflation, major economic events, large market shocks.
COMMON_MACRO_TERMS = [
    "RBI interest rate decision",
    "India inflation data",
    "India Union Budget",
    "India GDP growth",
    "Nifty Sensex market crash",
    "Sensex crash",
    "stock market crash India",
    "India stock market",
    "India government policy",
    "India tax changes",
]

# ---------------------------------------------------------------------------
# Per-company metadata
# ---------------------------------------------------------------------------
# symbol -> dict(ticker, sector, industry, subsector, aliases, low_confidence)
#
# `aliases` are the COMPANY-tier search phrases (project spec: "Apollo
# Hospitals", "Apollo Hospitals Enterprise", "APOLLOHOSP" for Apollo).
# The first alias is also used as the company's canonical display name
# for search/UI purposes if company_map.py's own name looks unhelpful
# (e.g. it strips a trailing "(Futures)" - see `display_alias` below).

COMPANY_METADATA = {
    "ADELc1_NS": dict(ticker="ADANIENT", sector="Diversified/Conglomerate",
                       industry="Diversified Conglomerate", subsector="Trading & Infrastructure",
                       aliases=["Adani Enterprises", "ADANIENT"]),
    "APLH": dict(ticker="APOLLOHOSP", sector="Healthcare",
                 industry="Hospitals & Healthcare Services", subsector="Multi-specialty Hospitals",
                 aliases=["Apollo Hospitals", "Apollo Hospitals Enterprise", "APOLLOHOSP"]),
    "APSEc1_NS": dict(ticker="ADANIPORTS", sector="Industrials",
                       industry="Marine Port & Logistics Services", subsector="Ports & SEZ",
                       aliases=["Adani Ports", "Adani Ports and SEZ", "ADANIPORTS"]),
    "ASPN": dict(ticker="ASIANPAINT", sector="Consumer Discretionary",
                 industry="Paints", subsector="Decorative Paints",
                 aliases=["Asian Paints", "ASIANPAINT"]),
    "AXBK": dict(ticker="AXISBANK", sector="Financial Services",
                 industry="Private Sector Bank", subsector="Banking",
                 aliases=["Axis Bank", "AXISBANK"]),
    "BAJAc1_NS": dict(ticker="BAJAJ-AUTO", sector="Automobile",
                       industry="Two & Three Wheelers", subsector="Motorcycles",
                       aliases=["Bajaj Auto", "BAJAJ-AUTO"]),
    "BAJE": dict(ticker="BAJAJFINSV", sector="Financial Services",
                 industry="Diversified Financial Services", subsector="Insurance & NBFC Holding",
                 aliases=["Bajaj Finserv", "BAJAJFINSV"]),
    "BJFN": dict(ticker="BAJFINANCE", sector="Financial Services",
                 industry="Non-Banking Financial Company", subsector="Consumer Finance",
                 aliases=["Bajaj Finance", "BAJFINANCE"]),
    "BJFS": dict(ticker="BAJAJFINSV", sector="Financial Services",
                 industry="Diversified Financial Services", subsector="Insurance & NBFC Holding",
                 aliases=["Bajaj Finserv", "BAJAJFINSV"]),
    "BRTI": dict(ticker="BHARTIARTL", sector="Telecommunications",
                 industry="Telecom Services", subsector="Wireless Telecom",
                 aliases=["Bharti Airtel", "Airtel", "BHARTIARTL"]),
    "CIPL": dict(ticker="CIPLA", sector="Healthcare",
                 industry="Pharmaceuticals", subsector="Generic Pharmaceuticals",
                 aliases=["Cipla", "CIPLA"]),
    "COAL": dict(ticker="COALINDIA", sector="Energy",
                 industry="Coal Mining", subsector="Coal Production",
                 aliases=["Coal India", "COALINDIA"]),
    "EICH": dict(ticker="EICHERMOT", sector="Automobile",
                 industry="Two Wheelers", subsector="Premium Motorcycles",
                 aliases=["Eicher Motors", "Royal Enfield", "EICHERMOT"]),
    "ETEA": dict(ticker="ETERNAL", sector="Consumer Services",
                 industry="Internet & Food Delivery", subsector="Online Food Delivery / Quick Commerce",
                 aliases=["Eternal", "Zomato", "ETERNAL"]),
    "GRAS": dict(ticker="GRASIM", sector="Materials",
                 industry="Cement & Chemicals", subsector="Diversified Materials",
                 aliases=["Grasim Industries", "GRASIM"]),
    "HALC": dict(ticker="HINDALCO", sector="Materials",
                 industry="Non-Ferrous Metals", subsector="Aluminium & Copper",
                 aliases=["Hindalco Industries", "HINDALCO"]),
    "HCLT": dict(ticker="HCLTECH", sector="Information Technology",
                 industry="IT Services", subsector="IT Consulting & Software",
                 aliases=["HCL Technologies", "HCLTECH"]),
    "HDBKc1_NS": dict(ticker="HDFCBANK", sector="Financial Services",
                       industry="Private Sector Bank", subsector="Banking",
                       aliases=["HDFC Bank", "HDFCBANK"]),
    "HDFLc1_NS": dict(ticker="HDFCLIFE", sector="Financial Services",
                       industry="Life Insurance", subsector="Insurance",
                       aliases=["HDFC Life Insurance", "HDFC Life", "HDFCLIFE"]),
    "HLL": dict(ticker="HINDUNILVR", sector="Consumer Staples",
                industry="FMCG", subsector="Household & Personal Products",
                aliases=["Hindustan Unilever", "HUL", "HINDUNILVR"]),
    "ICBK": dict(ticker="ICICIBANK", sector="Financial Services",
                 industry="Private Sector Bank", subsector="Banking",
                 aliases=["ICICI Bank", "ICICIBANK"]),
    "INFY": dict(ticker="INFY", sector="Information Technology",
                 industry="IT Services", subsector="IT Consulting & Software",
                 aliases=["Infosys", "INFY"]),
    "INGL": dict(ticker="IOC", sector="Energy",
                 industry="Oil & Gas Refining", subsector="Downstream Oil & Gas",
                 aliases=["Indian Oil Corporation", "IOC"], low_confidence=True),
    "ITC": dict(ticker="ITC", sector="Consumer Staples",
                industry="FMCG / Diversified", subsector="Tobacco, FMCG, Hotels & Agri",
                aliases=["ITC Limited", "ITC"]),
    "JIOF": dict(ticker="JIOFIN", sector="Financial Services",
                 industry="Non-Banking Financial Company", subsector="Digital Financial Services",
                 aliases=["Jio Financial Services", "JIOFIN"]),
    "JSTL": dict(ticker="JSWSTEEL", sector="Materials",
                 industry="Steel", subsector="Steel Production",
                 aliases=["JSW Steel", "JSWSTEEL"]),
    "KTKMc1_NS": dict(ticker="KOTAKBANK", sector="Financial Services",
                       industry="Private Sector Bank", subsector="Banking",
                       aliases=["Kotak Mahindra Bank", "KOTAKBANK"]),
    "LART": dict(ticker="LT", sector="Industrials",
                 industry="Engineering & Construction", subsector="Diversified Infrastructure",
                 aliases=["Larsen & Toubro", "L&T", "LT"]),
    "MAHM": dict(ticker="M&M", sector="Automobile",
                 industry="Passenger & Commercial Vehicles", subsector="SUVs & Tractors",
                 aliases=["Mahindra & Mahindra", "M&M"]),
    "MAXE": dict(ticker="MAXE", sector="Diversified/Conglomerate",
                 industry="Unconfirmed", subsector="Unconfirmed",
                 aliases=["Max"], low_confidence=True),
    "MRTI": dict(ticker="MARUTI", sector="Automobile",
                 industry="Passenger Vehicles", subsector="Cars & SUVs",
                 aliases=["Maruti Suzuki", "MARUTI"]),
    "NEST": dict(ticker="NESTLEIND", sector="Consumer Staples",
                 industry="FMCG", subsector="Packaged Foods",
                 aliases=["Nestle India", "NESTLEIND"]),
    "NTPCc1_NS": dict(ticker="NTPC", sector="Utilities",
                       industry="Power Generation", subsector="Thermal & Renewable Power",
                       aliases=["NTPC Limited", "NTPC"]),
    "ONGC": dict(ticker="ONGC", sector="Energy",
                 industry="Oil & Gas Exploration", subsector="Upstream Oil & Gas",
                 aliases=["Oil and Natural Gas Corporation", "ONGC"]),
    "PGRDc1_NS": dict(ticker="POWERGRID", sector="Utilities",
                       industry="Power Transmission", subsector="Electricity Transmission",
                       aliases=["Power Grid Corporation", "POWERGRID"]),
    "REDY": dict(ticker="DRREDDY", sector="Healthcare",
                 industry="Pharmaceuticals", subsector="Generic Pharmaceuticals",
                 aliases=["Dr Reddy's Laboratories", "Dr. Reddy's", "DRREDDY"]),
    "RELI": dict(ticker="RELIANCE", sector="Energy",
                 industry="Diversified Energy, Retail & Telecom", subsector="Diversified Conglomerate",
                 aliases=["Reliance Industries", "RELIANCE"]),
    "SBIc1_NS": dict(ticker="SBIN", sector="Financial Services",
                      industry="Public Sector Bank", subsector="Banking",
                      aliases=["State Bank of India", "SBI", "SBIN"]),
    "SHMF": dict(ticker="SHRIRAMFIN", sector="Financial Services",
                 industry="Non-Banking Financial Company", subsector="Vehicle & Retail Finance",
                 aliases=["Shriram Finance", "SHRIRAMFIN"]),
    "SUN": dict(ticker="SUNPHARMA", sector="Healthcare",
                industry="Pharmaceuticals", subsector="Specialty Pharmaceuticals",
                aliases=["Sun Pharmaceutical Industries", "Sun Pharma", "SUNPHARMA"]),
    "TACN": dict(ticker="TATACONSUM", sector="Consumer Staples",
                 industry="FMCG", subsector="Beverages & Packaged Foods",
                 aliases=["Tata Consumer Products", "TATACONSUM"]),
    "TAMO": dict(ticker="TATAMOTORS", sector="Automobile",
                 industry="Commercial & Passenger Vehicles", subsector="Cars, SUVs & Commercial Vehicles",
                 aliases=["Tata Motors", "TATAMOTORS"]),
    "TAMOc1_NS": dict(ticker="TATAMOTORS", sector="Automobile",
                       industry="Commercial & Passenger Vehicles", subsector="Cars, SUVs & Commercial Vehicles",
                       aliases=["Tata Motors", "TATAMOTORS"]),
    "TCSc1_NS": dict(ticker="TCS", sector="Information Technology",
                      industry="IT Services", subsector="IT Consulting & Software",
                      aliases=["Tata Consultancy Services", "TCS"]),
    "TEML": dict(ticker="TECHM", sector="Information Technology",
                 industry="IT Services", subsector="IT Consulting & Software",
                 aliases=["Tech Mahindra", "TECHM"]),
    "TISC": dict(ticker="TATASTEEL", sector="Materials",
                 industry="Steel", subsector="Steel Production",
                 aliases=["Tata Steel", "TATASTEEL"]),
    "TITN": dict(ticker="TITAN", sector="Consumer Discretionary",
                 industry="Jewellery & Watches", subsector="Branded Jewellery & Lifestyle",
                 aliases=["Titan Company", "TITAN"]),
    "TREN": dict(ticker="TRENT", sector="Consumer Discretionary",
                 industry="Retail", subsector="Fashion & Lifestyle Retail",
                 aliases=["Trent Limited", "TRENT"]),
    "ULTC": dict(ticker="ULTRACEMCO", sector="Materials",
                 industry="Cement", subsector="Cement Production",
                 aliases=["UltraTech Cement", "ULTRACEMCO"]),
    "WIPR": dict(ticker="WIPRO", sector="Information Technology",
                 industry="IT Services", subsector="IT Consulting & Software",
                 aliases=["Wipro", "WIPRO"]),
}


def get_metadata(symbol: str) -> dict:
    """
    Full metadata for `symbol` (one of src.data.company_map.CODE_MAP's
    keys), including the resolved sector_terms/macro_terms search lists
    - always returns something usable (falls back to a generic
    "Diversified/Conglomerate" bucket for a symbol with no explicit
    entry, rather than raising, so a newly-added company never breaks
    the news pipeline - it just gets the least-specific sector search
    until someone fills in real metadata for it here).
    """
    company_name, _ = CODE_MAP.get(symbol, (symbol, symbol))
    meta = COMPANY_METADATA.get(symbol)
    if meta is None:
        meta = dict(ticker=symbol, sector="Diversified/Conglomerate",
                    industry="Unclassified", subsector="Unclassified",
                    aliases=[company_name], low_confidence=True)

    sector = meta["sector"]
    sector_cfg = SECTOR_SEARCH_TERMS.get(sector, SECTOR_SEARCH_TERMS["Diversified/Conglomerate"])

    return {
        "symbol": symbol,
        "company_name": company_name,
        "ticker": meta["ticker"],
        "sector": sector,
        "industry": meta["industry"],
        "subsector": meta["subsector"],
        "aliases": meta["aliases"],
        "sector_terms": sector_cfg["sector_terms"],
        "macro_terms": sector_cfg["extra_macro_terms"] + COMMON_MACRO_TERMS,
        "low_confidence": meta.get("low_confidence", False),
    }


def display_alias(symbol: str) -> str:
    """The single best search/display phrase for `symbol` - the first
    (most natural-language) alias, e.g. 'Apollo Hospitals' rather than
    company_map.py's 'Apollo Hospitals' or a '(Futures)'-suffixed name."""
    return get_metadata(symbol)["aliases"][0]
