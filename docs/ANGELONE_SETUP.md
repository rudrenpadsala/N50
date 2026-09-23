# Connecting Angel One (SmartAPI) for live prices

This wires the "Live Market" page's `fetch_live_price()` up to real,
live NSE prices via Angel One's SmartAPI, instead of the "Live market
data unavailable" fallback.

**What was built for you (already in this repo, no further coding needed):**

| File | Purpose |
|---|---|
| `src/market/angelone_client.py` | Logs in to Angel One (API key + client code + PIN + TOTP), caches the session, downloads/caches Angel One's instrument list, and fetches the LTP (last traded price) for one symbol. |
| `src/market/angelone_symbol_map.py` | Maps this project's internal company codes (e.g. `"RELI"`) to the `(exchange, tradingsymbol)` pair Angel One needs (e.g. `("NSE", "RELIANCE-EQ")`). |
| `src/market/live_data.py` | Already updated to call Angel One automatically when it's configured (see below) — no changes needed here. |
| `.env.example` | Already has the Angel One variables listed — copy it to `.env` and fill in your own values. |
| `tests/test_angelone_client.py`, `tests/test_angelone_symbol_map.py` | Automated tests, all mocked (no real Angel One account or network access needed to run them). |

**What you still need to do yourself** (this is the part that has to
be done manually, since it needs your own Angel One account):

## Step 1 — Get SmartAPI access

1. You need an active Angel One trading/demat account.
2. Go to https://smartapi.angelone.in and log in with your Angel One credentials.
3. Click **"Create an app"**. Pick an app type that includes market data (e.g. "Trading APIs").
4. Angel One gives you an **API Key** — this is `ANGELONE_API_KEY`.

## Step 2 — Enable TOTP (required — plain-password API login is retired)

1. Go to https://smartapi.angelbroking.com/enable-totp
2. Log in with your client code and trading PIN, verify the OTP sent to your email/mobile.
3. You'll be shown a QR code. **Before scanning it with your authenticator app**, look for a
   "can't scan? enter this code manually" option — that text string is the **base32 TOTP
   secret** you need. This is `ANGELONE_TOTP_SECRET`. (If your app only shows the QR image, you
   can extract the secret from the QR code's `otpauth://...secret=XXXX` URL instead.)
4. Also add it to your regular authenticator app (Google Authenticator, Authy, etc.) as normal —
   you may want it there too for the Angel One website/app itself.

**Important:** `ANGELONE_TOTP_SECRET` is the *secret*, not a rotating 6-digit code. The code
generates a fresh 6-digit TOTP from this secret on every login automatically (via the `pyotp`
package) — you never need to type in a 6-digit code yourself.

## Step 3 — Gather your four credentials

| `.env` variable | Where it comes from |
|---|---|
| `ANGELONE_API_KEY` | Step 1 |
| `ANGELONE_CLIENT_CODE` | Your Angel One client ID (e.g. `A123456`) |
| `ANGELONE_PASSWORD` | Your Angel One **trading PIN** (not your website password) |
| `ANGELONE_TOTP_SECRET` | Step 2 |

## Step 4 — Fill in `.env`

```bash
cp .env.example .env
```

Then edit `.env` and fill in the four `ANGELONE_*` values above. Leave `MARKET_PROVIDER` blank —
`live_data.py` will pick Angel One automatically once `ANGELONE_API_KEY` is set. (Set
`MARKET_PROVIDER=angelone` explicitly only if you also configure the generic `MARKET_API_*`
variables and want to be sure Angel One wins.)

**Never commit your real `.env` file.** It contains a live trading-account PIN and TOTP secret —
treat it like a password.

## Step 5 — Verify your symbol mappings before trusting any price

`src/market/angelone_symbol_map.py` covers all 50 companies in the standard dataset:

- **37 equities** — mapped to an NSE cash-market `tradingsymbol` (e.g. `RELIANCE-EQ`).
- **11 futures-series codes** (`...c1_NS`) — mapped to the *underlying* name (e.g. `ADANIENT`)
  rather than a fixed contract symbol. `live_data.py` resolves the current **front-month**
  (nearest, not-yet-expired) NFO futures contract for that underlying automatically on every
  call, via `angelone_client.resolve_front_month_future()` — so this doesn't go stale when a
  contract expires and rolls over each month. Note a futures price is **not** the same instrument
  as the spot/EQ price and can differ meaningfully, especially near expiry.
- **2 codes left unmapped on purpose** (`INGL`, `MAXE`) — `company_map.py` itself flags these as
  "unverified underlying," so no live price is wired up for them until you confirm what company
  they actually represent.

These mappings are a best-effort match, not individually hand-verified against Angel One's live
data. Before you rely on any company's live price for a real decision, verify it resolves to a
real instrument:

```python
# Equities
from src.market.angelone_client import lookup_symbol_token
lookup_symbol_token("NSE", "RELIANCE-EQ")        # should return ('2885', None) or similar

# Futures
from src.market.angelone_client import resolve_front_month_future
resolve_front_month_future("ADANIENT")           # should return a contract dict, not an error
```

If either returns an error, fix that company's entry in `angelone_symbol_map.py` (correct the
`tradingsymbol`/underlying name, or leave it `None` if you don't want it live).

## Step 6 — Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the **Live Market** page for a mapped company (e.g. Reliance Industries). During NSE trading
hours (Mon–Fri, 09:15–15:30 IST) you should see a live price labeled **🔴 LIVE**; outside those
hours Angel One will still answer with the last traded price from the prior session.

## Notes, limits, and things worth knowing

- **Session lifetime:** Angel One sessions are valid until roughly midnight IST. This client
  re-logs-in automatically when the cached session goes stale or Angel One returns an
  auth-expired error — you don't need to restart anything.
- **Rate limits:** LTP calls are limited to 10/sec, 500/min, 5000/hour per Angel One's published
  limits; login is limited to about 1/sec. The Streamlit page's 30-second cache (`ttl=30` on
  `cached_live_price`) already keeps you well under this for normal use.
- **Instrument list caching:** The ~30 MB scrip master (symbol → token list) is downloaded once
  and cached to `data/cache/angelone_scrip_master.json` for up to 20 hours, so normal use doesn't
  re-download it on every price check.
- **This is a read-only integration.** It only fetches LTP quotes — it does not place orders,
  and this project (see `src/rl/`, `src/decision/`) does not execute trades automatically. Any
  BUY/SELL output remains a model suggestion for you to act on (or not) yourself.
- Angel One can change its API without notice; if login or price fetching starts failing, check
  https://smartapi.angelone.in/docs first, since the error message returned by
  `fetch_live_price()` will include whatever Angel One's API told us.
