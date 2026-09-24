"""
app.py

LIGHT FINTECH REDESIGN - a premium, light-themed visual identity for this
dashboard (clean cards, soft shadows, animated confidence ring, plain-
language "why" cards, collapsible advanced section) replacing the earlier
dark trading-terminal pass. Same page structure and business logic as
before (Stock Decision / Next Day Strategy / Live Market / Stock History /
About, plus a hidden Advanced Analysis page) - this pass only touches
presentation. Nothing in src/decision/engine.py, src/backtesting/,
src/risk/risk_engine.py, src/market/live_data.py, src/decision/next_day.py
or src/data/company_map.py is modified or called differently - this file
only reads from them, exactly as before.

What changed vs. the terminal pass:
  - Light background (soft off-white/blue-gray) with white cards, thin
    hairline borders and soft shadows instead of a dark charcoal-navy
    theme with left-edge accent bars.
  - Single font family (Inter) for both headings and body, with tabular
    numerals turned on for prices/percentages instead of a separate
    monospace family - a lighter, less "terminal" convention that still
    keeps numbers aligned.
  - The AI Decision card is now the unmistakable visual center of the
    page: bigger, centered, with an animated circular confidence ring
    that counts up from 0 on load (implemented as a small embedded
    HTML/CSS/JS component so it can actually animate, since Streamlit's
    own markdown can't run script tags reliably).
  - "Simple Market Indicators" is reframed as "Why is the AI saying
    {ACTION}?" - the same underlying engine.classify_* / RSI values,
    presented as short plain-language cards (icon, status, one-line
    explanation) instead of bare badges.
  - Raw technical figures (MA5, MA20, MACD, RSI value, individual model
    votes) now live inside a collapsible "Advanced Analysis" expander on
    every page, so a first-time user sees the decision and the why
    first, and can opt into the numbers.
  - A short animated "How the decision was reached" flow strip
    (Market Data -> Technical Signals -> News -> Risk -> AI Decision)
    is shown above the reasons list.
  - A minimal footer with a data-updated timestamp and the disclaimer
    is now shown once at the bottom of every page instead of only in
    the sidebar caption.
  - Company dropdown still shows real company names via the existing
    src/data/company_map.py (e.g. "Reliance Industries (RELI)"); to
    users this behaves like a searchable field since Streamlit's
    selectbox already filters options as you type.
  - Price chart still passes x="Date", y="Close" as plain columns (not a
    DatetimeIndex) for st.line_chart, which is the robust form.

No changes to src/decision/engine.py, src/backtesting/, or any
Sprint 1-5 training/backtesting code - this file only reads from them.

Run with:
    streamlit run app.py
"""

import os
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads a local .env file if present (see .env.example) - no-op if none exists
except ImportError:
    pass  # python-dotenv is optional; MARKET_API_* can also be set as real environment variables

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.decision import engine
from src.decision import next_day as next_day_module
from src.market import live_data
from src.risk import risk_engine
from src.backtesting.model_loader import ALGORITHMS
from src.data.company_map import company_for_code

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# ---------------------------------------------------------------------------
# Design tokens - signal colors are reserved EXCLUSIVELY for these meanings
# everywhere in the app (BUY/up = green, SELL/down = red, HOLD = amber).
# The indigo accent is reserved for "brand" / "this is live" moments only.
# ---------------------------------------------------------------------------

INK = "#12151C"
INK_DIM = "#6B7280"
INK_FAINT = "#9AA1AE"
BG = "#F5F6FA"
CARD = "#FFFFFF"
CARD_ALT = "#FAFBFD"
BORDER = "#E7E9F0"
GREEN = "#15A362"
RED = "#E0483F"
AMBER = "#D98A1A"
INDIGO = "#4F5BD5"

ACTION_COLORS = {"BUY": GREEN, "HOLD": AMBER, "SELL": RED, "AVOID": INK_DIM}
ACTION_BG = {"BUY": "rgba(21,163,98,0.10)", "HOLD": "rgba(217,138,26,0.10)", "SELL": "rgba(224,72,63,0.10)",
             "AVOID": "rgba(107,114,128,0.10)"}
ACTION_BADGE_COLORS = {name: (ACTION_COLORS[name], ACTION_BG[name]) for name in ACTION_COLORS}
BADGE_COLORS = {
    "UP": (GREEN, "rgba(21,163,98,0.10)"), "POSITIVE": (GREEN, "rgba(21,163,98,0.10)"), "LOW": (GREEN, "rgba(21,163,98,0.10)"),
    "DOWN": (RED, "rgba(224,72,63,0.10)"), "NEGATIVE": (RED, "rgba(224,72,63,0.10)"), "HIGH": (AMBER, "rgba(217,138,26,0.10)"),
    "SIDEWAYS": (INK_DIM, "rgba(107,114,128,0.10)"), "NEUTRAL": (INK_DIM, "rgba(107,114,128,0.10)"),
}
# Separate 4-point scale for the News & Risk Layer's overall/market/news
# risk levels (LOW/MODERATE/HIGH/EXTREME) - kept distinct from BADGE_COLORS
# above (whose "HIGH" means volatility-HIGH=amber) to avoid collapsing two
# different meanings of "HIGH" into the same color.
RISK_LEVEL_COLORS = {
    "LOW": (GREEN, "rgba(21,163,98,0.10)"), "NONE": (GREEN, "rgba(21,163,98,0.10)"),
    "MODERATE": (AMBER, "rgba(217,138,26,0.10)"),
    "HIGH": ("#E07A2E", "rgba(224,122,46,0.10)"),
    "EXTREME": (RED, "rgba(224,72,63,0.10)"), "CRITICAL": (RED, "rgba(224,72,63,0.10)"),
}
CONFIDENCE_COLORS = {
    "High": (GREEN, "rgba(21,163,98,0.10)"), "Medium": (AMBER, "rgba(217,138,26,0.10)"),
    "Low": (INK_DIM, "rgba(107,114,128,0.10)"),
}
TIME_RANGES = {"1 Month": 30, "3 Months": 90, "6 Months": 182, "1 Year": 365, "All Available Data": None}
ACTION_ID_TO_NAME = {0: "HOLD", 1: "BUY", 2: "SELL"}
ACTION_ICONS = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "AVOID": "⚪"}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

:root {
    --ink: #12151C; --ink-dim: #6B7280; --ink-faint: #9AA1AE;
    --bg: #F5F6FA; --card: #FFFFFF; --card-alt: #FAFBFD; --border: #E7E9F0;
    --green: #15A362; --red: #E0483F; --amber: #D98A1A; --indigo: #4F5BD5;
}

/* BUG FIX (2026-09): the old rule was `header[data-testid="stHeader"]
   {visibility: hidden;}`. visibility:hidden is inherited by children
   unless a child explicitly overrides it - and on mobile, the sidebar's
   open/close toggle (the arrow/hamburger control) lives inside this
   same header element. Hiding the whole header therefore also hid the
   only way to open the sidebar on a phone, even though the sidebar
   itself still worked fine on desktop (where it's always expanded and
   never needs that toggle). Fix: hide only the top-right icon toolbar
   (Streamlit's own menu / GitHub / Deploy icons) and make the header's
   background transparent for the same clean look, instead of hiding
   the header element itself - and explicitly force every known variant
   of the sidebar-toggle control to stay visible as a safety net, since
   the exact element Streamlit uses for it has changed across versions. */
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent; box-shadow: none;}
header[data-testid="stHeader"] [data-testid="stToolbar"] {visibility: hidden; height: 0;}
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important; display: flex !important; opacity: 1 !important;
}
.stApp {background: var(--bg);}
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1000px;}

.stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, .stApp li {
    font-family: 'Inter', sans-serif; color: var(--ink);
    font-feature-settings: 'tnum' 1; font-variant-numeric: tabular-nums;
}
h1, h2, h3, h4, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Inter', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em;
    color: var(--ink);
}

/* Sidebar */
[data-testid="stSidebar"] {background: var(--card); border-right: 1px solid var(--border);}
[data-testid="stSidebar"] * {font-family: 'Inter', sans-serif;}

/* Hero header */
.hero {
    padding: 4px 0 22px 0; margin-bottom: 6px;
    animation: fadeInUp 0.5s ease both;
}
.hero .eyebrow {
    display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem; font-weight: 600;
    color: var(--indigo); background: rgba(79,91,213,0.08); padding: 4px 10px; border-radius: 999px;
    letter-spacing: 0.02em; margin-bottom: 10px;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif !important; font-size: 2.1rem !important; margin: 0 0 4px 0 !important;
}
.hero .subtitle {color: var(--ink-dim); font-size: 1rem; margin: 0;}

/* Status strip - live/historical provenance, always in the same place */
.status-strip {
    display: flex; align-items: flex-start; gap: 10px; background: var(--card-alt);
    border: 1px solid var(--border); border-radius: 10px; padding: 11px 15px;
    margin-bottom: 1.1rem; font-size: 0.86rem; color: var(--ink-dim); line-height: 1.5;
    animation: fadeInUp 0.4s ease both;
}
.status-strip .dot {width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 5px;}
.status-strip .dot.live {background: var(--green); box-shadow: 0 0 0 3px rgba(21,163,98,0.15); animation: pulse 1.8s ease-in-out infinite;}
.status-strip .dot.historical {background: var(--ink-faint);}
.status-strip strong {color: var(--ink); font-weight: 600;}
.status-strip code {
    background: rgba(18,21,28,0.05); border-radius: 4px; padding: 1px 6px;
    font-size: 0.85em; color: var(--indigo);
}
@keyframes pulse {
    0%, 100% {box-shadow: 0 0 0 0 rgba(21,163,98,0.45);}
    50% {box-shadow: 0 0 0 6px rgba(21,163,98,0);}
}
@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(8px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Cards */
.card-row {display: flex; gap: 12px; margin-bottom: 1.1rem; flex-wrap: wrap;}
.info-card {
    flex: 1 1 0; min-width: 150px; background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(16,24,40,0.03);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
    animation: fadeInUp 0.35s ease both;
}
.info-card:hover {box-shadow: 0 6px 16px rgba(16,24,40,0.07); transform: translateY(-1px);}
.info-card .label {font-size: 0.72rem; color: var(--ink-dim); font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;}
.info-card .value {font-size: 1.3rem; color: var(--ink); font-weight: 700; margin-top: 5px; line-height: 1.25;}
.info-card .sub {font-size: 0.82rem; font-weight: 600; margin-top: 3px;}

.badge {
    display: inline-block; padding: 3px 11px; border-radius: 999px; font-weight: 600; font-size: 0.85rem;
}

/* Signal / "why" cards */
.signal-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.03); transition: box-shadow 0.15s ease, transform 0.15s ease;
    height: 100%; animation: fadeInUp 0.4s ease both;
}
.signal-card:hover {box-shadow: 0 6px 16px rgba(16,24,40,0.07); transform: translateY(-1px);}
.signal-card .signal-icon {font-size: 1.3rem;}
.signal-card .signal-title {font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; color: var(--ink-dim); text-transform: uppercase; margin-top: 6px;}
.signal-card .signal-status {font-size: 1.05rem; font-weight: 700; margin-top: 2px;}
.signal-card .signal-text {font-size: 0.85rem; color: var(--ink-dim); margin-top: 5px; line-height: 1.4;}
.signal-bar-track {height: 5px; border-radius: 999px; background: rgba(18,21,28,0.06); margin-top: 10px; overflow: hidden;}
.signal-bar-fill {height: 100%; border-radius: 999px; animation: growBar 0.9s cubic-bezier(.2,.8,.2,1) both;}
@keyframes growBar {from {width: 0%;} }

/* Hero AI Decision card - the single bold element per page */
.decision-wrap {
    background: linear-gradient(180deg, var(--card) 0%, var(--card-alt) 100%);
    border: 1px solid var(--border); border-radius: 20px; padding: 30px 24px 24px 24px;
    text-align: center; margin-bottom: 1.3rem; box-shadow: 0 8px 24px rgba(16,24,40,0.06);
    animation: fadeInUp 0.5s ease both;
}
.decision-wrap .kicker {font-size: 0.78rem; color: var(--ink-dim); font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;}
.decision-wrap .action {
    font-family: 'Space Grotesk', sans-serif; font-size: 3rem; font-weight: 700; margin: 4px 0 2px 0;
    letter-spacing: -0.02em;
}
.decision-wrap .message {font-size: 0.98rem; color: var(--ink-dim); margin-bottom: 4px;}
.decision-wrap .updated {font-size: 0.78rem; color: var(--ink-faint); margin-top: 10px;}

.reason-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px; animation: fadeInUp 0.3s ease both;
}
.reason-card .reason-title {font-weight: 600; color: var(--ink); font-size: 0.94rem;}
.reason-card .reason-text {color: var(--ink-dim); font-size: 0.88rem; margin-top: 2px;}

.summary-card {background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px;}
.summary-card .row {display: flex; justify-content: space-between; padding: 7px 0; font-size: 0.9rem; border-bottom: 1px solid var(--border);}
.summary-card .row:last-child {border-bottom: none;}
.summary-card .row .k {color: var(--ink-dim); font-weight: 500;}
.summary-card .row .v {color: var(--ink); font-weight: 700;}

/* "How the decision was reached" flow strip */
.flow-strip {display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin: 4px 0 1.1rem 0;}
.flow-step {
    background: var(--card); border: 1px solid var(--border); border-radius: 999px; padding: 6px 14px;
    font-size: 0.82rem; font-weight: 600; color: var(--ink-dim); animation: fadeInUp 0.4s ease both;
}
.flow-arrow {color: var(--ink-faint); font-size: 0.9rem;}

.disclaimer {color: var(--ink-faint); font-size: 0.8rem; margin-top: 0.6rem; line-height: 1.5;}

/* Footer */
.app-footer {
    margin-top: 2.2rem; padding-top: 16px; border-top: 1px solid var(--border);
    color: var(--ink-faint); font-size: 0.8rem; line-height: 1.6;
}
.app-footer .foot-title {color: var(--ink-dim); font-weight: 700; font-size: 0.85rem;}

/* Native Streamlit widgets - align to the light palette */
.stButton>button {
    font-family: 'Inter', sans-serif; font-weight: 600; border-radius: 10px;
    border: 1px solid var(--border); background: var(--card); color: var(--ink);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.stButton>button:hover {box-shadow: 0 4px 10px rgba(16,24,40,0.08); transform: translateY(-1px); border-color: var(--indigo);}
.stButton>button[kind="primary"] {background: var(--indigo); color: #fff; border-color: var(--indigo);}
[data-testid="stMetricValue"] {font-feature-settings: 'tnum' 1;}
.stCaption, [data-testid="stCaptionContainer"] {color: var(--ink-faint) !important;}
[data-testid="stExpander"] {border: 1px solid var(--border); border-radius: 12px; background: var(--card);}

/* ===========================================================================
   MOBILE RESPONSIVENESS
   Streamlit already serves a correct responsive viewport meta tag and
   stacks st.columns() vertically below ~640px on its own - the rules
   below handle everything this app's OWN custom markup (card grids, the
   hero header, the decision card, the flow strip, buttons) doesn't get
   for free from Streamlit. Two breakpoints: 640px (tablets / large
   phones landscape) and 480px (phones portrait, the common case).
   =========================================================================== */

/* Applies at every width, not just mobile - prevents long company names,
   headlines or reasons from ever forcing horizontal scroll on a narrow
   screen, which is the single most common way a "desktop" page breaks
   on mobile. */
.stApp, .stMarkdown, .info-card .value, .decision-wrap, .reason-card,
.summary-card .row, .status-strip, .hero h1, .hero .subtitle {
    overflow-wrap: break-word; word-break: break-word;
}
html, body {overflow-x: hidden;}

@media (max-width: 640px) {
    .block-container {padding-top: 1.1rem; padding-left: 0.9rem; padding-right: 0.9rem; padding-bottom: 2.2rem;}

    .hero {padding: 2px 0 16px 0;}
    .hero h1 {font-size: clamp(1.5rem, 6vw, 1.9rem) !important; line-height: 1.15;}
    .hero .subtitle {font-size: 0.88rem;}
    .hero .eyebrow {font-size: 0.72rem; padding: 3px 9px; margin-bottom: 8px;}

    /* Card grids: let 2 narrower cards sit side by side instead of the
       desktop min-width forcing an awkward 1-then-2 wrap. */
    .card-row {gap: 8px; margin-bottom: 0.9rem;}
    .info-card {min-width: 122px; padding: 11px 12px; border-radius: 10px;}
    .info-card .label {font-size: 0.66rem;}
    .info-card .value {font-size: 1.08rem;}
    .info-card .sub {font-size: 0.76rem;}

    .signal-card {padding: 13px;}
    .signal-card .signal-status {font-size: 0.98rem;}
    .signal-card .signal-text {font-size: 0.8rem;}

    /* The AI Decision card is the visual centerpiece - keep it bold but
       scale the giant action text down so BUY/HOLD/SELL never wraps or
       crowds the card edges on a narrow phone. */
    .decision-wrap {padding: 22px 16px 18px 16px; border-radius: 16px;}
    .decision-wrap .action {font-size: clamp(2.1rem, 12vw, 2.8rem);}
    .decision-wrap .message {font-size: 0.9rem;}

    .status-strip {padding: 10px 12px; font-size: 0.8rem; gap: 8px;}
    .reason-card {padding: 10px 13px;}
    .reason-card .reason-title {font-size: 0.88rem;}
    .reason-card .reason-text {font-size: 0.82rem;}

    .summary-card {padding: 13px 14px;}
    .summary-card .row {font-size: 0.84rem; padding: 6px 0;}

    /* Flow strip (Market Data -> Technical Signals -> ... -> AI Decision):
       5 pills + arrows never fit a phone's width, and letting them wrap
       breaks the left-to-right "flow" reading order. Instead, keep it on
       one line and let it scroll horizontally - a well-understood mobile
       pattern - rather than reflowing into a confusing multi-row wrap. */
    .flow-strip {
        flex-wrap: nowrap; overflow-x: auto; -webkit-overflow-scrolling: touch;
        padding-bottom: 6px; margin-left: -0.9rem; margin-right: -0.9rem;
        padding-left: 0.9rem; padding-right: 0.9rem; scrollbar-width: thin;
    }
    .flow-strip::-webkit-scrollbar {height: 4px;}
    .flow-strip::-webkit-scrollbar-thumb {background: var(--border); border-radius: 999px;}
    .flow-step {flex-shrink: 0; font-size: 0.76rem; padding: 5px 12px; white-space: nowrap;}

    .app-footer {margin-top: 1.6rem; padding-top: 12px; font-size: 0.76rem;}

    /* Touch targets: Apple/Google's own guidance is a 44px minimum hit
       area - the desktop button height was comfortably mouse-sized but
       tight for a thumb. Also spans full width on mobile, which is the
       standard "primary action" pattern on phones and easier to hit
       than a small button floating on the left. */
    .stButton>button {min-height: 44px; width: 100%; font-size: 0.92rem;}

    /* Native Streamlit inputs/selects: match the same touch-friendly
       minimum height so the company picker and price field are as easy
       to tap as everything else on the page. */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    .stTextInput>div>div>input {min-height: 44px;}

    [data-testid="stExpander"] summary {padding: 10px 12px; font-size: 0.9rem;}

    /* Safety net for Streamlit's own st.columns() (the 3-up Why-signal
       cards, Upside/Downside metrics, and the confidence-ring centering
       columns): Streamlit stacks these on its own past its default
       breakpoint, but this guarantees a clean full-width single column
       with sane spacing even if that default ever changes, rather than
       silently depending on undocumented framework behavior. */
    [data-testid="stHorizontalBlock"] {flex-wrap: wrap !important; row-gap: 10px;}
    [data-testid="stHorizontalBlock"] > div {min-width: 100% !important; flex: 1 1 100% !important;}
    [data-testid="stMetricValue"] {font-size: 1.3rem !important;}
}

@media (max-width: 480px) {
    /* Below ~480px (most phones in portrait) a 2-up card grid still
       feels cramped for 3-4 metric cards - go fully single-column so
       every number stays legible instead of shrinking further. */
    .info-card {flex-basis: 100%; min-width: 0;}
    .decision-wrap .action {font-size: clamp(1.9rem, 14vw, 2.4rem);}
    .hero h1 {font-size: clamp(1.35rem, 7vw, 1.6rem) !important;}
}

/* Landscape phones (short viewport height): the sticky/animated
   elements shouldn't eat too much vertical space when there's little
   of it to begin with. */
@media (max-height: 420px) and (orientation: landscape) {
    .hero {padding: 2px 0 10px 0;}
    .decision-wrap {padding: 16px 16px 14px 16px;}
}
</style>
"""


# ---------------------------------------------------------------------------
# Company display names (reuses Sprint 1's src/data/company_map.py)
# ---------------------------------------------------------------------------

def display_name(symbol: str) -> str:
    name, _ = company_for_code(symbol)
    if name.startswith("UNKNOWN") or name.startswith("UNVERIFIED"):
        return symbol
    return name


def company_label(symbol: str) -> str:
    name = display_name(symbol)
    return f"{name} ({symbol})" if name != symbol else symbol


# ---------------------------------------------------------------------------
# Cached data access (re-reads only when the underlying files change)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_companies() -> list:
    companies = engine.list_companies()
    return sorted(companies, key=company_label)


@st.cache_data(show_spinner=False, ttl=900)
def cached_decision(company: str) -> dict:
    # BUG FIX (2026-09): was ttl=300 (5 min). This is the call that hits
    # Angel One's historical-candle endpoint - the endpoint you've been
    # seeing "exceeding access rate" 403s from. Daily candle data doesn't
    # go stale in 5 minutes, so a short TTL bought no real freshness but
    # tripled how often every stock you look at re-hits that endpoint
    # when browsing between several stocks in a session. 15 minutes cuts
    # that call volume by 3x with no meaningful loss of freshness for a
    # daily-bar-based decision.
    return engine.get_decision(company)


@st.cache_data(show_spinner=False, ttl=300)
def cached_risk_assessment(company: str, company_display_name: str, decision: dict, _news_refresh_nonce: int = 0):
    """
    News & Risk Layer evaluation (src/risk/risk_engine.py), cached
    separately from the RL decision itself so a 'Refresh News' click
    doesn't also re-trigger a live market fetch, and vice versa. The
    already-fetched `decision` is passed straight through to
    risk_engine.evaluate() so this never issues a second market-data
    request. `_news_refresh_nonce` is bumped by the Refresh News button
    to force a fresh news fetch (bypassing both this Streamlit cache and
    src/news/news_api.py's own internal cache).
    """
    from src.news import news_api
    if _news_refresh_nonce:
        news_api.clear_cache()
    return risk_engine.evaluate(company, company_display_name, rl_decision=decision)


@st.cache_data(show_spinner=False)
def cached_historical_decisions(company: str, num_days: int) -> pd.DataFrame:
    return engine.historical_decisions(company, num_days)


@st.cache_data(show_spinner=False)
def cached_backtest_summary(company: str):
    return engine.backtest_summary(company)


@st.cache_data(show_spinner=False)
def cached_price_history(company: str) -> pd.DataFrame:
    path = os.path.join(PROJECT_ROOT, "data", "features", f"{company}_features.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")


@st.cache_data(show_spinner=False)
def cached_algorithm_comparison() -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, "algorithm_comparison.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def cached_final_results(company: str) -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, "final_results.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["Company"] == company]


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------

def info_card(label: str, value: str, sub: str = None, sub_color: str = None, neutral: bool = True) -> str:
    sub_html = f'<div class="sub" style="color:{sub_color or "var(--ink-dim)"};">{sub}</div>' if sub else ""
    return f'<div class="info-card"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>'

def badge_card(label: str, value: str, color_map: dict = None) -> str:
    color, bg = (color_map or BADGE_COLORS).get(value, (INK_DIM, "rgba(107,114,128,0.10)"))
    return (
        f'<div class="info-card"><div class="label">{label}</div>'
        f'<div style="margin-top:6px;"><span class="badge" style="color:{color};background:{bg};">{value}</span></div></div>'
    )

def render_cards(cards_html: list) -> None:
    st.markdown(f'<div class="card-row">{"".join(cards_html)}</div>', unsafe_allow_html=True)


def render_data_source_banner(result: dict) -> None:
    """
    Persistent, compact "live vs. historical" status strip shown at the
    top of the Stock Decision and Next Day Strategy pages - same spot,
    same visual language every time, so a user is never left guessing
    whether a number is real-time. Explicitly calls out when the live
    instrument is a FUTURES contract (not the plain equity/spot price)
    - futures and spot prices are different instruments and can
    legitimately differ by a percent or more, which otherwise looks
    like a bug when compared against a spot-price source like Google.
    """
    if result.get("data_source") == "live":
        info = result.get("instrument_info") or {}
        as_of = result["latest_date"].strftime("%d %b %Y")
        if info.get("is_future"):
            expiry = info.get("expiry")
            expiry_str = expiry.strftime("%d %b %Y") if hasattr(expiry, "strftime") else str(expiry)
            body = (
                f'<div class="dot live"></div><div><strong>Live</strong> · futures contract '
                f'<code>{info.get("tradingsymbol", "?")}</code> (NFO, expiry {expiry_str}) via Angel One, '
                f'as of {as_of}. This is the <strong>futures price</strong>, not the NSE spot/equity price — '
                f'the two normally differ by up to ~1% (the futures "basis"), so it won\'t exactly match a '
                f'spot-price source like Google Finance.</div>'
            )
        else:
            tradingsymbol = info.get("tradingsymbol")
            suffix = f' · {info.get("exchange", "NSE")}: <code>{tradingsymbol}</code>' if tradingsymbol else ""
            body = f'<div class="dot live"></div><div><strong>Live</strong>{suffix} via Angel One, as of {as_of}.</div>'
    else:
        reason = result.get("live_unavailable_reason")
        reason_html = f" ({reason})" if reason else ""
        body = (f'<div class="dot historical"></div><div><strong>Historical dataset</strong> — live market data '
                f'is unavailable right now{reason_html}.</div>')
    st.markdown(f'<div class="status-strip">{body}</div>', unsafe_allow_html=True)


def render_confidence_ring(percent: float, color: str, label: str) -> None:
    """
    Animated circular confidence ring that counts up from 0 to `percent`
    on load. Implemented as a small self-contained HTML/CSS/JS component
    (via st.components.v1.html) rather than st.markdown, because
    Streamlit's markdown does not reliably execute <script> tags, and a
    genuine count-up / stroke animation needs JS to drive it.
    """
    pct = max(0, min(100, round(percent)))
    size, stroke = 132, 10
    radius = (size - stroke) / 2
    circumference = 2 * 3.14159265 * radius
    html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                font-family:'Inter',sans-serif;">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="#E7E9F0" stroke-width="{stroke}"/>
        <circle id="ring" cx="{size/2}" cy="{size/2}" r="{radius}" fill="none" stroke="{color}"
                stroke-width="{stroke}" stroke-linecap="round"
                stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{circumference:.2f}"
                transform="rotate(-90 {size/2} {size/2})"
                style="transition: stroke-dashoffset 1.1s cubic-bezier(.2,.8,.2,1);"/>
        <text id="pct" x="50%" y="47%" text-anchor="middle" dominant-baseline="middle"
              font-size="26" font-weight="700" fill="#12151C" font-family="Inter, sans-serif">0%</text>
        <text x="50%" y="65%" text-anchor="middle" dominant-baseline="middle"
              font-size="11" font-weight="600" fill="#6B7280" font-family="Inter, sans-serif"
              letter-spacing="0.03em">{label.upper()}</text>
      </svg>
    </div>
    <script>
      (function() {{
        const target = {pct};
        const circumference = {circumference:.2f};
        const ring = document.getElementById('ring');
        const pctText = document.getElementById('pct');
        setTimeout(function() {{
          const offset = circumference - (target / 100) * circumference;
          ring.style.strokeDashoffset = offset;
        }}, 60);
        let current = 0;
        const steps = 40;
        const stepTime = 1100 / steps;
        const increment = target / steps;
        const timer = setInterval(function() {{
          current += increment;
          if (current >= target) {{ current = target; clearInterval(timer); }}
          pctText.textContent = Math.round(current) + '%';
        }}, stepTime);
      }})();
    </script>
    """
    components.html(html, height=size + 12)


def render_flow_strip() -> None:
    steps = ["Market Data", "Technical Signals", "Market News", "Risk Analysis", "AI Decision"]
    parts = []
    for i, step in enumerate(steps):
        delay = f"style=\"animation-delay:{i * 0.08:.2f}s;\""
        parts.append(f'<div class="flow-step" {delay}>{step}</div>')
        if i < len(steps) - 1:
            parts.append('<div class="flow-arrow">→</div>')
    st.markdown(f'<div class="flow-strip">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_footer(as_of_label: str = None) -> None:
    updated = as_of_label or datetime.now().strftime("%d %b %Y, %H:%M")
    st.markdown(
        f"""
        <div class="app-footer">
            <div class="foot-title">AI Stock Decision Center</div>
            <div>Market insights powered by data and AI.</div>
            <div style="margin-top:4px;">Data last refreshed: {updated}</div>
            <div class="disclaimer">This application provides analytical insights and model-based predictions for
            informational purposes only, using historical backtesting and virtual money. It is not financial advice,
            does not guarantee future returns, and does not execute real trades.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hours_ago_text(hours: float) -> str:
    if hours < 1:
        return "Just now"
    if hours < 24:
        return f"{int(round(hours))} hour{'s' if round(hours) != 1 else ''} ago"
    days = hours / 24.0
    return f"{int(round(days))} day{'s' if round(days) != 1 else ''} ago"


# ---------------------------------------------------------------------------
# Plain-language "why" signal cards, replacing raw badges as the primary
# view. The underlying values are the exact same engine.classify_*() /
# RSI outputs as before - only the presentation is friendlier.
# ---------------------------------------------------------------------------

SIGNAL_COPY = {
    "Price Trend": {
        "UP": ("📈", GREEN, "Positive", "Price is trading above its short-term average."),
        "DOWN": ("📉", RED, "Negative", "Price is trading below its short-term average."),
        "SIDEWAYS": ("➡️", INK_DIM, "Flat", "Price is moving without a clear short-term direction."),
    },
    "Momentum": {
        "POSITIVE": ("📊", GREEN, "Strong", "Recent price movement shows positive momentum."),
        "NEGATIVE": ("📊", RED, "Weak", "Recent price movement shows negative momentum."),
        "NEUTRAL": ("📊", INK_DIM, "Neutral", "Recent price movement shows little clear momentum."),
    },
    "Volatility": {
        "LOW": ("⚡", GREEN, "Calm", "Price swings are currently smaller than usual."),
        "HIGH": ("⚡", AMBER, "Elevated", "Price swings are currently larger than usual."),
        "NEUTRAL": ("⚡", INK_DIM, "Typical", "Price swings are currently in a typical range."),
    },
}


def render_why_signals(decision: dict) -> None:
    st.markdown(f"#### Why is the AI saying {ACTION_ID_TO_NAME[decision['action']]}?")
    trend = engine.classify_price_trend(decision["trend"])
    momentum = engine.classify_momentum(decision["rsi_value"])
    volatility = engine.classify_volatility_simple(decision["volatility_condition"])
    rsi_value = decision["rsi_value"]

    cols = st.columns(3)
    for col, (title, key) in zip(cols, [("Price Trend", trend), ("Momentum", momentum), ("Volatility", volatility)]):
        icon, color, status, text = SIGNAL_COPY[title].get(key, ("•", INK_DIM, key, ""))
        bar_pct = {GREEN: 82, AMBER: 55, RED: 30, INK_DIM: 50}.get(color, 50)
        with col:
            st.markdown(
                f"""
                <div class="signal-card">
                    <div class="signal-icon">{icon}</div>
                    <div class="signal-title">{title}</div>
                    <div class="signal-status" style="color:{color};">{status}</div>
                    <div class="signal-text">{text}</div>
                    <div class="signal-bar-track"><div class="signal-bar-fill" style="width:{bar_pct}%;background:{color};"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    rsi_icon = "⚡"
    rsi_status, rsi_text = ("Healthy", "Momentum is not currently in an extreme zone.")
    rsi_color = GREEN
    if rsi_value >= 70:
        rsi_status, rsi_text, rsi_color = "Overbought", "Momentum is running hot - a pullback is possible.", AMBER
    elif rsi_value <= 30:
        rsi_status, rsi_text, rsi_color = "Oversold", "Momentum is running cold - a bounce is possible.", AMBER
    st.markdown(
        f"""
        <div class="signal-card" style="margin-top:2px;">
            <div class="signal-icon">{rsi_icon}</div>
            <div class="signal-title">RSI ({rsi_value:.0f})</div>
            <div class="signal-status" style="color:{rsi_color};">{rsi_status}</div>
            <div class="signal-text">{rsi_text}</div>
            <div class="signal-bar-track"><div class="signal-bar-fill" style="width:{min(100, rsi_value):.0f}%;background:{rsi_color};"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


SECTOR_ICONS = {
    "Healthcare": "🏥", "Financial Services": "🏦", "Information Technology": "💻",
    "Automobile": "🚗", "Materials": "⛏️", "Energy": "⛽", "Consumer Staples": "🛒",
    "Consumer Discretionary": "🛍️", "Industrials": "🏗️", "Utilities": "💡",
    "Telecommunications": "📡", "Diversified/Conglomerate": "🏢", "Consumer Services": "📱",
}


def _render_news_section(title: str, icon: str, analysis) -> None:
    """One of the three news sections (project spec: 'Show three
    separate sections... Display only the top 2-3 relevant articles in
    each'). `analysis` is a src.news.news_analyzer.NewsAnalysis for
    that single tier."""
    st.markdown(f"##### {icon} {title}")
    if not analysis.available:
        st.caption(f"Data unavailable: {analysis.reason}")
        return
    if not analysis.articles:
        st.caption("No relevant recent articles found for this tier.")
        return
    for article in analysis.articles:
        color, bg = BADGE_COLORS.get(article.sentiment_label, (INK_DIM, "rgba(107,114,128,0.10)"))
        st.markdown(
            f'<div class="reason-card">'
            f'<span class="badge" style="color:{color};background:{bg};font-size:0.78rem;">{article.sentiment_label}</span> '
            f'<a href="{article.url}" target="_blank" style="color:{INK};text-decoration:none;font-weight:600;">{article.headline}</a>'
            f'<div class="reason-text">{article.source} · {_hours_ago_text(article.hours_ago)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_risk_layer(assessment, key_prefix: str) -> None:
    """
    Shared News & Risk Layer section (project spec #16-#18): RL Signal /
    News Sentiment / Risk Level / Final Decision cards, plain-language
    reasons, THREE news sections (Company/Sector/Macro), and a "Risk
    Analysis" breakdown. Used on the Stock Decision, Live Market, and
    Next Day Strategy pages so the presentation is identical everywhere.
    `assessment` is a src.risk.risk_engine.RiskAssessment.
    """
    rl_action_name = ACTION_ID_TO_NAME[assessment.rl_decision["action"]]
    final_action = assessment.final_action
    tiered = assessment.tiered_news
    tiered_risk = assessment.tiered_news_risk
    guard = assessment.guard

    st.markdown("#### Market News & Risk")

    if st.button("🔄 Refresh News", key=f"{key_prefix}_refresh_news"):
        nonce_key = f"{key_prefix}_news_refresh_nonce"
        st.session_state[nonce_key] = st.session_state.get(nonce_key, 0) + 1

    company_sentiment = (tiered.company.aggregated_sentiment_label
                          if (tiered.company.available and tiered.company.articles) else "NEUTRAL")
    render_cards([
        badge_card("RL Signal", rl_action_name, color_map=ACTION_BADGE_COLORS),
        badge_card("Company News", company_sentiment),
        badge_card("Risk Level", assessment.overall_risk_level, color_map=RISK_LEVEL_COLORS),
        badge_card("Final Decision", final_action, color_map=ACTION_BADGE_COLORS),
    ])
    if not tiered.company.available:
        st.caption(f"📰 Company news data unavailable ({tiered.company.reason}) - the final decision above is "
                    f"based on the RL signal, sector/macro news, and market risk only.")

    if guard.overridden:
        risk_color = RISK_LEVEL_COLORS.get(assessment.overall_risk_level, (RED, "rgba(224,72,63,0.10)"))[0]
        st.markdown(
            f'<div class="reason-card" style="border-left:3px solid {risk_color};">'
            f'<div class="reason-title">⚠ Risk Alert</div>'
            f'<div class="reason-text">{guard.reason}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("##### Why?")
    sector_icon = SECTOR_ICONS.get(tiered.sector, "🏭")
    company_label = (tiered.company.aggregated_sentiment_label
                      if (tiered.company.available and tiered.company.articles) else "No data")
    sector_label = (tiered.sector_news.aggregated_sentiment_label
                     if (tiered.sector_news.available and tiered.sector_news.articles) else "No data")
    macro_label = (tiered.macro.aggregated_sentiment_label
                    if (tiered.macro.available and tiered.macro.articles) else "No data")
    st.markdown(f"- 🏢 **Company News ({tiered.company_name}):** {company_label}")
    st.markdown(f"- {sector_icon} **{tiered.sector} Sector:** {sector_label}")
    st.markdown(f"- 🌐 **Market/Macro:** {macro_label}")
    st.markdown(f"- 📊 **Market Risk:** {assessment.market.market_risk_label if assessment.market.available else 'N/A'}")
    for reason_text in assessment.reasons:
        st.markdown(f"- {reason_text}")

    st.markdown(
        f'<div class="disclaimer" style="margin-top:8px;">'
        f'Model Agreement: {guard.model_agreement_pct:.0f}% (this is how many of the RL models agree, '
        f'risk-adjusted - not a probability of profit).</div>',
        unsafe_allow_html=True,
    )

    st.markdown("##### Latest News")
    _render_news_section(f"Company News ({tiered.company_name})", "🏢", tiered.company)
    _render_news_section(f"{tiered.sector} Sector News", sector_icon, tiered.sector_news)
    _render_news_section("Market & Economic News", "🌐", tiered.macro)

    st.markdown("##### Risk Breakdown")
    market = assessment.market
    render_cards([
        badge_card("Company Risk", tiered_risk.company.news_risk_label if tiered_risk.company.available else "N/A",
                   color_map=RISK_LEVEL_COLORS),
        badge_card("Sector Risk", tiered_risk.sector.news_risk_label if tiered_risk.sector.available else "N/A",
                   color_map=RISK_LEVEL_COLORS),
        badge_card("Macro Risk", tiered_risk.macro.news_risk_label if tiered_risk.macro.available else "N/A",
                   color_map=RISK_LEVEL_COLORS),
        badge_card("Market Risk", market.market_risk_label if market.available else "N/A", color_map=RISK_LEVEL_COLORS),
    ])
    render_cards([
        badge_card("Overall News Risk", tiered_risk.overall_news_risk_label, color_map=RISK_LEVEL_COLORS),
        badge_card("Overall Risk", assessment.overall_risk_level, color_map=RISK_LEVEL_COLORS),
        info_card("Weights (C/S/M)", f"{int(tiered.weights_used['company']*100)}/"
                  f"{int(tiered.weights_used['sector']*100)}/{int(tiered.weights_used['macro']*100)}%",
                  "configurable in config.yaml"),
        info_card("Event Severity", str(max(
            tiered.company.top_event_severity if tiered.company.available else 0,
            tiered.sector_news.top_event_severity if tiered.sector_news.available else 0,
            tiered.macro.top_event_severity if tiered.macro.available else 0,
        )), "0=None … 4=Critical"),
    ])
    explanation_bits = []
    if tiered_risk.reasons:
        explanation_bits.append(tiered_risk.reasons[0])
    if market.available and market.reasons:
        explanation_bits.append(market.reasons[0])
    if explanation_bits:
        st.caption(" ".join(explanation_bits))
    else:
        st.caption("News and market conditions currently look normal.")


def render_decision_card(action_name: str, confidence_label: str, confidence_pct: float = None) -> None:
    color = ACTION_COLORS.get(action_name, INK)
    message = {
        "BUY": "Strong positive signal.",
        "HOLD": "No strong edge either way right now.",
        "SELL": "Strong negative signal.",
    }.get(action_name, "")
    st.markdown(
        f"""
        <div class="decision-wrap">
            <div class="kicker">AI Decision</div>
            <div class="action" style="color:{color};">{action_name}</div>
            <div class="message">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if confidence_pct is not None:
        c1, c2, c3 = st.columns([1, 1.1, 1])
        with c2:
            render_confidence_ring(confidence_pct, color, f"{confidence_label} confidence")
    else:
        conf_color, conf_bg = CONFIDENCE_COLORS.get(confidence_label, (INK_DIM, "rgba(107,114,128,0.10)"))
        st.markdown(
            f'<div style="text-align:center;margin-top:-8px;">'
            f'<span class="badge" style="color:{conf_color};background:{conf_bg};">Confidence: {confidence_label}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="text-align:center;" class="disclaimer">Updated {datetime.now().strftime("%H:%M")} · '
        f'model-based estimate, not a guarantee.</div>',
        unsafe_allow_html=True,
    )


def render_price_chart(df: pd.DataFrame) -> None:
    """Robust line chart: pass Date/Close as explicit plain columns
    (never a DatetimeIndex) - the indexed form intermittently rendered
    a blank chart (Vega-Lite 'Infinite extent' warning) on first paint."""
    chart_df = df[["Date", "Close"]].dropna().sort_values("Date")
    if chart_df.empty:
        st.info("No price history available for this range.")
        return
    st.line_chart(chart_df, x="Date", y="Close", use_container_width=True, color=INDIGO)

def filter_by_range(df: pd.DataFrame, range_label: str) -> pd.DataFrame:
    days = TIME_RANGES[range_label]
    if days is None or df.empty:
        return df
    cutoff = df["Date"].max() - pd.Timedelta(days=days)
    return df[df["Date"] >= cutoff]


# ---------------------------------------------------------------------------
# Page: Stock Decision (default / "Overview")
# ---------------------------------------------------------------------------

def render_stock_decision(company: str) -> None:
    st.title("AI Stock Decision")
    st.caption("Understand the market. Analyze the trend. Make informed decisions.")

    try:
        with st.spinner("Loading the latest decision…"):
            decision = cached_decision(company)
    except engine.DecisionError as exc:
        st.warning(str(exc))
        return

    prev_close = decision["previous_close"]
    has_prev_close = prev_close is not None and not pd.isna(prev_close)
    change_pct = (decision["latest_price"] - prev_close) / prev_close * 100 if has_prev_close and prev_close != 0 else None

    change_amount = decision["latest_price"] - prev_close if has_prev_close else None
    change_str, change_color = "—", None
    if change_pct is not None:
        change_str = f"₹{change_amount:+,.2f}  ({change_pct:+.2f}%)"
        change_color = GREEN if change_pct >= 0 else RED

    render_data_source_banner(decision)

    render_cards([
        info_card("Latest Available Price", f"₹{decision['latest_price']:,.2f}",
                   sub=change_str, sub_color=change_color),
        info_card("Latest Available Data Date", decision["latest_date"].strftime("%d %b %Y")),
        info_card("Previous Close", f"₹{prev_close:,.2f}" if has_prev_close else "—"),
    ])

    action_name = ACTION_ID_TO_NAME[decision["action"]]
    render_decision_card(action_name, decision["confidence_label"], decision.get("confidence_pct"))
    if decision.get("weak_signal"):
        st.caption(
            "⚠️ The internal models show little preference between actions for this stock right now "
            "— treat this decision with extra caution."
        )

    st.markdown("#### How the decision was reached")
    render_flow_strip()

    render_why_signals(decision)

    st.markdown("##### The details")
    for icon, title, text in engine.build_reasons(decision):
        st.markdown(
            f'<div class="reason-card"><div class="reason-title">{icon} {title}</div>'
            f'<div class="reason-text">{text}</div></div>',
            unsafe_allow_html=True,
        )

    news_nonce = st.session_state.get("stockdecision_news_refresh_nonce", 0)
    assessment = cached_risk_assessment(company, display_name(company), decision, news_nonce)
    render_risk_layer(assessment, key_prefix="stockdecision")

    st.markdown("#### Stock Price History")
    price_df = cached_price_history(company)
    range_label = st.select_slider("Time range", options=list(TIME_RANGES.keys()), value="6 Months")
    render_price_chart(filter_by_range(price_df, range_label))

    with st.expander("📐 Advanced Analysis — technical indicators & raw figures"):
        st.markdown("###### Technical Indicators")
        macd = decision.get("macd")
        macd_signal = decision.get("macd_signal")
        macd_sub = None
        if macd is not None and macd_signal is not None:
            macd_sub = "Above signal" if macd > macd_signal else "Below signal"
        render_cards([
            info_card("MA5", f"₹{decision['ma5']:,.2f}"),
            info_card("MA20", f"₹{decision['ma20']:,.2f}"),
            info_card("RSI", f"{decision['rsi_value']:.0f}", engine.rsi_text(decision["rsi_value"])),
            info_card("MACD", f"{macd:+.2f}" if macd is not None else "—",
                       sub=macd_sub, sub_color=GREEN if macd_sub == "Above signal" else RED),
        ])
        if macd is None:
            st.caption("MACD needs recent live daily candles to compute and isn't available for a historical-only decision.")

        st.markdown("###### Decision Summary")
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="row"><span class="k">Company</span><span class="v">{company_label(company)}</span></div>
                <div class="row"><span class="k">Latest Available Price</span><span class="v">₹{decision['latest_price']:,.2f}</span></div>
                <div class="row"><span class="k">Latest Data Date</span><span class="v">{decision['latest_date'].strftime('%d %b %Y')}</span></div>
                <div class="row"><span class="k">AI Decision</span><span class="v" style="color:{ACTION_COLORS[action_name]};">{action_name}</span></div>
                <div class="row"><span class="k">Confidence</span><span class="v">{decision['confidence_label']}</span></div>
                <div class="row"><span class="k">Price Trend</span><span class="v">{engine.classify_price_trend(decision['trend'])}</span></div>
                <div class="row"><span class="k">Momentum</span><span class="v">{engine.classify_momentum(decision['rsi_value'])}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer(decision["latest_date"].strftime("%d %b %Y"))


# ---------------------------------------------------------------------------
# Shared warning banner for the two prediction/live pages
# ---------------------------------------------------------------------------

AI_ANALYSIS_WARNING = (
    "⚠️ AI-generated analysis is for educational/research purposes only. "
    "It does not guarantee future returns and does not execute real trades."
)

DIRECTION_ICONS = {"UP": "📈", "DOWN": "📉", "SIDEWAYS": "➡️"}


# ---------------------------------------------------------------------------
# Page: Next Day Strategy
# ---------------------------------------------------------------------------

def render_next_day_strategy(company: str) -> None:
    st.title("Next Market Day")
    st.caption(
        "\"What should I do for the next trading day?\" - enter the price you expect "
        "to buy or enter this stock at."
    )

    try:
        latest_row = engine.get_latest_feature_row(company)
        default_price = float(latest_row["Close"])
    except engine.DecisionError:
        default_price = 0.0

    raw_price = st.text_input(
        "Reference Price (₹) - the price you expect to buy/enter at",
        value=f"{default_price:.2f}" if default_price else "",
    )
    predict_clicked = st.button("Predict Next Day Strategy", type="primary")

    if not predict_clicked:
        st.caption(AI_ANALYSIS_WARNING)
        return

    entry_price, error = next_day_module.validate_price_input(raw_price)
    if error:
        st.error(error)
        st.caption(AI_ANALYSIS_WARNING)
        return

    try:
        with st.spinner("Running the next-day model…"):
            result = next_day_module.next_day_strategy(company, entry_price)
    except engine.DecisionError as exc:
        st.warning(str(exc))
        st.caption(AI_ANALYSIS_WARNING)
        return

    action_name = ACTION_ID_TO_NAME[result["action"]]
    direction = engine.classify_price_trend(result["trend"])
    direction_icon = DIRECTION_ICONS.get(direction, "➡️")
    price_range = result["price_range"]

    render_data_source_banner(result)

    st.markdown(
        f"""
        <div class="decision-wrap" style="text-align:left;padding:20px 24px;">
            <div class="kicker">Next day strategy</div>
            <div style="font-size:1.1rem;font-weight:700;color:{INK};margin-top:2px;">{company_label(company)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_cards([
        info_card("Entry Price", f"₹{entry_price:,.2f}"),
        info_card("Expected Direction", f"{direction_icon} {direction}"),
        info_card(
            "Estimated Price Range",
            f"₹{price_range['low']:,.2f} – ₹{price_range['high']:,.2f}" if price_range else "Not available",
        ),
    ])

    render_decision_card(action_name, result["confidence_label"], result.get("confidence_pct"))
    if result.get("weak_signal"):
        st.caption(
            "⚠️ The internal models show little preference between actions for this stock right now "
            "— treat this decision with extra caution."
        )
    st.caption("This is a model-based estimate, not a guaranteed future price.")

    if price_range and result["upside_pct"] is not None:
        st.markdown("#### Price Scenario")
        up_col, down_col = st.columns(2)
        up_col.metric("Potential Upside", f"{result['upside_pct']:+.2f}%")
        down_col.metric("Potential Downside", f"{result['downside_pct']:+.2f}%")
        st.caption(
            f"Estimated range ₹{price_range['low']:,.2f} – ₹{price_range['high']:,.2f}, "
            f"based on this stock's recent historical volatility "
            f"(±{price_range['volatility_pct']:.2f}% around the latest available price of "
            f"₹{price_range['basis_price']:,.2f}). This is a statistical estimate, NOT a guaranteed target."
        )
    else:
        st.info(
            "An estimated price range isn't available for this company yet (not enough recent "
            "price history to compute a reliable volatility estimate)."
        )

    st.markdown("#### Why this strategy?")
    for icon, title, text in engine.build_reasons(result):
        st.markdown(
            f'<div class="reason-card"><div class="reason-title">{icon} {title}</div>'
            f'<div class="reason-text">{text}</div></div>',
            unsafe_allow_html=True,
        )

    news_nonce = st.session_state.get("nextday_news_refresh_nonce", 0)
    assessment = cached_risk_assessment(company, display_name(company), result, news_nonce)
    render_risk_layer(assessment, key_prefix="nextday")

    st.markdown(f"**{action_name}** favored by the current model analysis and the price you entered.")
    st.caption(AI_ANALYSIS_WARNING)
    render_footer()


# ---------------------------------------------------------------------------
# Page: Live Market
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=30)
def cached_live_price(company: str, _refresh_nonce: int = 0):
    return live_data.fetch_live_price(company)


def render_live_market(company: str) -> None:
    st.title("Live Market")
    st.caption("\"What is happening with this stock TODAY?\"")

    if "live_refresh_nonce" not in st.session_state:
        st.session_state["live_refresh_nonce"] = 0
    if st.button("Refresh"):
        st.session_state["live_refresh_nonce"] += 1

    status = live_data.market_status()
    market_dot = "live" if status["is_open"] else "historical"
    st.markdown(
        f'<div class="status-strip"><div class="dot {market_dot}"></div>'
        f'<div><strong>Market {"open" if status["is_open"] else "closed"}</strong> · {status["label"]} '
        f'(as of {status["now_ist"].strftime("%H:%M:%S")} IST — weekday/trading-hours check only, '
        f'does not account for exchange holidays)</div></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Fetching live price…"):
        live = cached_live_price(company, st.session_state["live_refresh_nonce"])

    try:
        latest_row = engine.get_latest_feature_row(company)
        historical_price = float(latest_row["Close"])
        historical_date = pd.Timestamp(latest_row["Date"])
        static_previous_close = engine.get_previous_close(company, latest_row["Date"])
    except engine.DecisionError as exc:
        st.warning(str(exc))
        return

    # A real "yesterday's close" to compare a LIVE price against - reused
    # from engine.get_decision()'s own live computation (which fetches it
    # from live Angel One candles) rather than the static historical file,
    # whose last row can be weeks old. Comparing a live price against that
    # stale static close would produce a nonsensical "today's change".
    try:
        decision = cached_decision(company)
        decision_error = None
    except engine.DecisionError as exc:
        decision = None
        decision_error = str(exc)

    live_previous_close = None
    if decision is not None and decision.get("data_source") == "live":
        pc = decision.get("previous_close")
        if pc is not None and not pd.isna(pc):
            live_previous_close = float(pc)

    # BUG FIX (2026-09): Angel One's own historical-candle endpoint has a
    # long-standing, currently unresolved bug on their platform that
    # falsely rejects requests as rate-limited even far below their
    # documented limits (see angelone_client.py's AngelOneQuote/get_ltp
    # comments) - which is what made "today's change" unavailable here
    # essentially every time, even though the live price itself (a
    # different, unaffected endpoint) was working fine. Angel One's LTP
    # response already includes open/high/low/previous-close alongside
    # the price in the SAME call, so prefer those - already fetched
    # above as `live` - over the historical-endpoint-derived values,
    # and only fall back to the latter (or to nothing) if the LTP
    # response happened not to include them.
    if live.available and live.previous_close is not None:
        live_previous_close = live.previous_close

    if live.available:
        current_price = live.price
        price_label = "Live price"
        last_updated = live.timestamp.strftime("%H:%M:%S IST") if live.timestamp else "—"
        if live_previous_close is not None:
            previous_close = live_previous_close
            change = current_price - previous_close
            change_pct = (change / previous_close * 100.0) if previous_close else None
        else:
            # We have a live price but no live previous-close to compare it
            # against yet - show it plainly rather than fabricate a change
            # figure against the (possibly weeks-old) static dataset.
            previous_close = float("nan")
            change = None
            change_pct = None
            reason = (decision or {}).get("live_unavailable_reason")
            suffix = f" ({reason})." if reason else "."
            st.caption("Today's change isn't shown - a live previous-close reference isn't available yet" + suffix)
    else:
        st.markdown(
            f'<div class="status-strip"><div class="dot historical"></div>'
            f'<div><strong>Live data unavailable</strong> — {live.reason or "showing the latest available price instead."}</div></div>',
            unsafe_allow_html=True,
        )
        current_price = historical_price
        previous_close = static_previous_close
        change = historical_price - previous_close if not pd.isna(previous_close) else None
        change_pct = (change / previous_close * 100.0) if change is not None and previous_close else None
        price_label = "Last available price"
        last_updated = f"{historical_date.strftime('%d %b %Y')} (historical, not live)"

    change_str = f"₹{change:+,.2f}" if change is not None else None
    change_pct_str = f"{change_pct:+.2f}%" if change_pct is not None else None
    change_sub = f"{change_str}  ({change_pct_str})" if change_str and change_pct_str else "—"
    change_color = GREEN if (change or 0) >= 0 else RED

    render_cards([
        info_card(price_label, f"₹{current_price:,.2f}", sub=change_sub, sub_color=change_color),
        info_card("Previous Close", f"₹{previous_close:,.2f}" if not pd.isna(previous_close) else "—"),
    ])

    # BUG FIX (2026-09): same reasoning as previous_close above - prefer
    # today's high/low from the working LTP response over the
    # historical-endpoint-derived decision fields, which are usually
    # unavailable because of Angel One's rate-limit bug on that endpoint.
    today_high = live.high if (live.available and live.high is not None) else (decision or {}).get("today_high")
    today_low = live.low if (live.available and live.low is not None) else (decision or {}).get("today_low")
    volume = (decision or {}).get("volume")
    volume_change_pct = (decision or {}).get("volume_change_pct")
    volume_sub = f"{volume_change_pct:+.1f}% vs prev. day" if volume_change_pct is not None else None
    render_cards([
        info_card("Today's High", f"₹{today_high:,.2f}" if today_high is not None else "—"),
        info_card("Today's Low", f"₹{today_low:,.2f}" if today_low is not None else "—"),
        info_card("Volume", f"{volume:,.0f}" if volume is not None else "—",
                   sub=volume_sub, sub_color=GREEN if (volume_change_pct or 0) >= 0 else RED),
    ])
    st.caption(f"Last Updated: {last_updated}")

    if decision is None:
        st.warning(decision_error or "No trained models available for this company yet.")
        return

    st.markdown("#### AI Decision — Today")
    render_data_source_banner(decision)
    action_name = ACTION_ID_TO_NAME[decision["action"]]
    render_decision_card(action_name, decision["confidence_label"], decision.get("confidence_pct"))

    render_why_signals(decision)

    news_nonce = st.session_state.get("livemarket_news_refresh_nonce", 0)
    assessment = cached_risk_assessment(company, display_name(company), decision, news_nonce)
    render_risk_layer(assessment, key_prefix="livemarket")

    with st.expander("📐 Advanced Analysis — technical indicators & raw figures"):
        macd = decision.get("macd")
        macd_signal = decision.get("macd_signal")
        macd_sub = ("Above signal" if macd > macd_signal else "Below signal") if (macd is not None and macd_signal is not None) else None
        render_cards([
            info_card("MA5", f"₹{decision['ma5']:,.2f}"),
            info_card("MA20", f"₹{decision['ma20']:,.2f}"),
            info_card("RSI", f"{decision['rsi_value']:.0f}", engine.rsi_text(decision["rsi_value"])),
            info_card("MACD", f"{macd:+.2f}" if macd is not None else "—",
                       sub=macd_sub, sub_color=GREEN if macd_sub == "Above signal" else RED),
        ])

    st.markdown("#### Today's Price Chart")
    if live.available:
        st.info("Intraday data isn't available from the configured live market API - showing the latest available daily prices instead.")
    else:
        st.info("Live market data unavailable - showing the latest available daily prices instead.")
    price_df = cached_price_history(company)
    render_price_chart(filter_by_range(price_df, "1 Month"))

    st.caption(AI_ANALYSIS_WARNING)
    render_footer()


# ---------------------------------------------------------------------------
# Page: Stock History
# ---------------------------------------------------------------------------

def render_stock_history(company: str) -> None:
    st.title("Historical Performance")
    st.markdown(f"**Company:** {company_label(company)}")

    st.markdown("#### Price History")
    price_df = cached_price_history(company)
    range_label = st.select_slider("Time range", options=list(TIME_RANGES.keys()),
                                    value="1 Year", key="history_range")
    render_price_chart(filter_by_range(price_df, range_label))

    st.markdown("#### Historical AI Decisions")
    st.caption(
        "Illustrative only - shows what the AI would have suggested on each recent day, "
        "assuming a fresh look each time (not an audited trading simulation)."
    )
    hist = cached_historical_decisions(company, 30)
    if hist.empty:
        st.info("No trained models available yet for this company.")
    else:
        display = hist.copy()
        display["Date"] = display["Date"].dt.strftime("%d %b %Y")
        display["Price"] = display["Price"].map(lambda p: f"₹{p:,.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("#### How the AI Performed Historically")
    st.caption("Historical simulation only - not a prediction of future returns.")
    summary = cached_backtest_summary(company)
    if summary is None:
        st.info("Backtest results not yet available for this company. Run the backtest pipeline first.")
    else:
        render_cards([
            info_card("Starting Virtual Money", f"₹{summary['starting_capital']:,.0f}"),
            info_card("Ending Virtual Money", f"₹{summary['ending_value']:,.0f}"),
            info_card("Historical Return", f"{summary['return_pct']:+.2f}%",
                      sub_color=GREEN if summary["return_pct"] >= 0 else RED),
            info_card("Maximum Loss During Test", f"{summary['max_drawdown_pct']:.2f}%"),
        ])
        st.caption(f"Averaged across {summary['num_algorithms']} internal AI models.")

    render_footer()


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def render_about() -> None:
    st.title("About This Project")
    st.markdown(
        "This system uses Reinforcement Learning to analyze historical stock market data "
        "and generate BUY, HOLD, or SELL decisions."
    )
    st.markdown("#### How it works")
    render_flow_strip()
    st.markdown(
        "1. Historical market data is collected.\n"
        "2. Market features are calculated.\n"
        "3. The AI learns from a simulated trading environment.\n"
        "4. Multiple AI methods analyze the situation internally.\n"
        "5. The system combines the analysis.\n"
        "6. One final BUY, HOLD, or SELL decision is shown to you."
    )
    st.info(
        "This project uses virtual money and historical data for evaluation. "
        "It does not execute real trades."
    )
    render_footer()


# ---------------------------------------------------------------------------
# Page: Advanced Analysis (hidden - only reachable via the sidebar
# "Show Advanced Analysis" checkbox, off by default). Same content and
# data as the previous "Developer Analysis" page - renamed to match the
# "keep advanced info under Advanced" navigation pattern used elsewhere.
# ---------------------------------------------------------------------------

def render_developer_analysis(company: str) -> None:
    st.title("Advanced Analysis")
    st.caption("Technical detail for the 5 internal RL algorithms. Not shown to normal users by default.")

    try:
        decision = cached_decision(company)
    except engine.DecisionError as exc:
        st.warning(str(exc))
        return

    st.markdown("#### Individual algorithm votes (current state)")
    vote_rows = []
    for algo in ALGORITHMS:
        if algo in decision["actions_by_algorithm"]:
            action_id = decision["actions_by_algorithm"][algo]
            q_row = decision["q_rows_by_algorithm"][algo]
            margin = engine.action_margin(q_row)
            vote_rows.append({
                "Algorithm": algo,
                "Action": ACTION_ID_TO_NAME[action_id],
                "Q(HOLD)": round(float(q_row[0]), 4),
                "Q(BUY)": round(float(q_row[1]), 4),
                "Q(SELL)": round(float(q_row[2]), 4),
                "Top-2 Margin": round(margin, 5),
                "Weak (<0.001)": "⚠️" if margin < engine.MARGIN_THRESHOLD else "",
            })
        else:
            vote_rows.append({"Algorithm": algo, "Action": "not trained",
                               "Q(HOLD)": None, "Q(BUY)": None, "Q(SELL)": None,
                               "Top-2 Margin": None, "Weak (<0.001)": None})
    st.dataframe(pd.DataFrame(vote_rows), use_container_width=True, hide_index=True)

    st.markdown(
        f"**Final decision:** {ACTION_ID_TO_NAME[decision['action']]}  \n"
        f"**Votes:** HOLD={decision['votes'][0]}, BUY={decision['votes'][1]}, SELL={decision['votes'][2]}  \n"
        f"**Confidence:** {decision['confidence_pct']:.1f}% ({decision['confidence_label']})  \n"
        f"**Tie-break applied:** {decision['tie_broken']}  \n"
        f"**Avg. winning-vote margin:** {decision['avg_winning_margin']:.5f} "
        f"({'below' if decision['weak_signal'] else 'at or above'} the {engine.MARGIN_THRESHOLD} threshold "
        f"— {'confidence downgraded' if decision['weak_signal'] else 'no downgrade'})"
    )

    st.markdown("#### Sprint 5 backtest metrics for this company")
    company_results = cached_final_results(company)
    if company_results.empty:
        st.info("No backtest results available for this company yet.")
    else:
        cols = ["Algorithm", "Total Return %", "Annualized Return %", "Sharpe Ratio",
                "Maximum Drawdown %", "Win Rate %", "Number of Trades"]
        st.dataframe(company_results[cols].round(2), use_container_width=True, hide_index=True)

    st.markdown("#### Average performance across all backtested companies")
    comparison_df = cached_algorithm_comparison()
    if comparison_df.empty:
        st.info("Run scripts/run_backtest.py to populate the algorithm comparison report.")
    else:
        st.dataframe(comparison_df.round(2), use_container_width=True, hide_index=True)
        rl_only = comparison_df[comparison_df["Algorithm"] != "Buy & Hold"]
        if not rl_only.empty:
            best_return = rl_only.loc[rl_only["Average Return %"].idxmax()]
            best_sharpe = rl_only.loc[rl_only["Average Sharpe Ratio"].idxmax()]
            c1, c2 = st.columns(2)
            c1.metric("Best by Return (avg.)", best_return["Algorithm"], f"{best_return['Average Return %']:.2f}%")
            c2.metric("Best by Sharpe Ratio (avg.)", best_sharpe["Algorithm"], f"{best_sharpe['Average Sharpe Ratio']:.2f}")

    render_footer()


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Stock Decision Center", layout="centered")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

companies = cached_companies()

st.sidebar.markdown(
    '<div style="display:flex;align-items:center;gap:8px;padding:4px 0 14px 0;">'
    '<div style="width:9px;height:9px;border-radius:50%;background:#4F5BD5;"></div>'
    '<span style="font-family:\'Inter\',sans-serif;font-weight:800;font-size:1.02rem;color:#12151C;">'
    'AI Stock Decision Center</span></div>',
    unsafe_allow_html=True,
)
if not companies:
    st.sidebar.error("No company data found.")
    st.error("Market data is temporarily unavailable.\n\nPlease try again in a moment.")
    st.stop()

st.sidebar.caption("🔍 Search company or ticker — type to filter the list below.")
selected_company = st.sidebar.selectbox("Company", companies, format_func=company_label, label_visibility="collapsed")

st.sidebar.markdown("---")
pages = ["🏠 Stock Decision", "🔮 Next Day Strategy", "⚡ Live Market", "📊 Stock History", "ℹ️ About"]

developer_mode = st.sidebar.checkbox(
    "Show Advanced Analysis", value=False,
    help="Shows the 5 individual RL algorithms and technical backtest metrics, kept out of the way by default.",
)
if developer_mode:
    pages.append("🛠️ Advanced Analysis")

page = st.sidebar.radio("Navigate", pages, label_visibility="collapsed")
page = page.split(" ", 1)[1] if " " in page else page  # strip the icon prefix for routing

st.sidebar.markdown("---")
st.sidebar.caption(
    "This project uses historical backtesting and virtual money. "
    "It does not provide guaranteed future returns and does not execute real trades."
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">● AI-POWERED</div>
        <h1>AI Stock Decision Center</h1>
        <p class="subtitle">Understand the market. Analyze the trend. Make informed decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Stock Decision":
    render_stock_decision(selected_company)
elif page == "Next Day Strategy":
    render_next_day_strategy(selected_company)
elif page == "Live Market":
    render_live_market(selected_company)
elif page == "Stock History":
    render_stock_history(selected_company)
elif page == "About":
    render_about()
elif page == "Advanced Analysis":
    render_developer_analysis(selected_company)