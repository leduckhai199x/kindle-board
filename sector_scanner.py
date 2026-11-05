"""
Fidelity Sector Scanner v2
--------------------------
Step 1-3: Sector table — all 11 Fidelity sector ETFs, latest price vs
          yesterday's close and 3 months ago, color-coded green/red.
Step 4:   Deep scan — top-5 holdings of each sector ETF, each analyzed on:
            • financials straight from SEC EDGAR — the 10-K/10-Q filings
              themselves (10 years of annual data for Compounder, latest
              quarter + trailing 12 months for Dip-buyer); Yahoo only fills gaps
            • analyst ratings + price target
            • MACD (12/26/9) and Fibonacci position in the 6-month range
            • events: next earnings date, recent 8-K filings (red-flag filings
              block a green dot), political/geopolitical headlines
          Each stock gets a short description and a 🟢/🔴 health dot.

Run it:
    pip install streamlit yfinance pandas plotly
    streamlit run sector_scanner.py

SEC asks every app to identify itself with a name + email: the app asks once
and saves it in sec_contact.txt (on Streamlit Cloud, add SEC_CONTACT to the
app's Secrets instead). SEC data is saved in the sec_cache/ folder.

Note: automated screening signals from public data — not buy/sell advice.
"""

import datetime as dt
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# All 11 GICS sectors via Fidelity's sector ETF lineup
SECTOR_ETFS = {
    "FTEC": "Information Technology",
    "FCOM": "Communication Services",
    "FDIS": "Consumer Discretionary",
    "FSTA": "Consumer Staples",
    "FENY": "Energy",
    "FNCL": "Financials",
    "FHLC": "Health Care",
    "FIDU": "Industrials",
    "FMAT": "Materials",
    "FREL": "Real Estate",
    "FUTY": "Utilities",
}

COLUMNS = ["Ticker", "Sector", "Price", "Prev Close", "Day %", "3-Mo %", "1-Yr %", "5-Yr %"]

# When Yahoo blocks its ETF-holdings data, the app rebuilds each top-5 itself.
# These sector funds are market-cap weighted, so their biggest holdings are
# simply the biggest companies in the sector: the candidates below get ranked
# by LIVE market cap (share count x price — Yahoo endpoints that keep working
# when fund data is blocked). Listed in rough size order; that order is only
# used if live market caps are unavailable too. If a new giant emerges in a
# sector, add its ticker here.
SECTOR_CANDIDATES = {
    "FTEC": ["NVDA", "AAPL", "MSFT", "AVGO", "ORCL", "MU", "AMD", "PLTR",
             "CSCO", "IBM", "CRM", "AMAT", "LRCX", "INTU"],
    "FCOM": ["META", "GOOGL", "GOOG", "NFLX", "TMUS", "DIS", "T", "VZ",
             "CMCSA", "TTWO", "CHTR"],          # EA left: taken private (Aug 2026)
    "FDIS": ["AMZN", "TSLA", "HD", "MCD", "BKNG", "TJX", "LOW", "SBUX",
             "NKE", "ABNB", "ORLY", "RCL", "MAR", "CMG"],
    "FSTA": ["WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL",
             "MNST", "TGT", "KDP"],
    "FENY": ["XOM", "CVX", "COP", "WMB", "EOG", "KMI", "OKE", "MPC", "PSX",
             "SLB", "VLO", "OXY", "BKR"],
    "FNCL": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP",
             "C", "SCHW", "BLK", "SPGI", "PGR"],
    "FHLC": ["LLY", "JNJ", "ABBV", "UNH", "MRK", "ABT", "TMO", "ISRG",
             "AMGN", "PFE", "BSX", "DHR", "GILD", "SYK", "VRTX"],
    "FIDU": ["GE", "CAT", "RTX", "GEV", "UBER", "HON", "BA", "UNP", "ETN",
             "DE", "LMT", "ADP", "PH", "TT"],
    "FMAT": ["LIN", "SHW", "NEM", "ECL", "APD", "FCX", "CTVA", "NUE",
             "DD", "VMC", "MLM", "DOW"],
    "FREL": ["PLD", "AMT", "WELL", "EQIX", "SPG", "O", "PSA", "DLR",
             "CCI", "CBRE", "VICI", "EXR"],
    "FUTY": ["NEE", "SO", "CEG", "DUK", "VST", "AEP", "SRE", "D", "EXC",
             "PCG", "XEL", "PEG", "ED"],
}
FALLBACK_HOLDINGS = {etf: names[:5] for etf, names in SECTOR_CANDIDATES.items()}

# Last successful live holdings get saved here, so a short Yahoo outage falls
# back to real recent lists (with fund weights) before estimating.
HOLDINGS_CACHE_FILE = Path(__file__).with_name("holdings_cache.json")
HOLDINGS_CACHE_MAX_AGE_DAYS = 14
AUTH_ERROR_MARKERS = ("401", "unauthorized", "invalid crumb", "unable to access")

GREEN = "background-color: #c6efce; color: #006100; font-weight: 600"
RED = "background-color: #ffc7ce; color: #9c0006; font-weight: 600"

GREEN_DOT, RED_DOT, GRAY_DOT = "🟢", "🔴", "⚪"
MIN_SCORE_FOR_GREEN = 2  # net signal score needed for a green dot

POLITICAL_KEYWORDS = [
    "tariff", "sanction", "regulat", "antitrust", "lawsuit", "election",
    "congress", "senate", "white house", "president", "fed ", "federal reserve",
    "china", "taiwan", "russia", "ukraine", "war", "opec", "subsid", "ban",
    "probe", "investigation", "fda", "doj", "ftc", "sec ", "export control",
    "national security", "geopolit", "supreme court", "epa",
]


# ═══════════════════════ PART A: sector table (steps 1-3) ═══════════════════════

def fetch_history(tickers: list[str], period: str = "6mo") -> pd.DataFrame:
    """Daily closes, one column per ticker."""
    data = yf.download(
        tickers, period=period, interval="1d",
        auto_adjust=False, progress=False, threads=True,
    )
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close


def get_live_price(ticker: str) -> float | None:
    try:
        price = float(yf.Ticker(ticker).fast_info["lastPrice"])
        return price if price > 0 else None
    except Exception:
        return None


def current_and_prev(s: pd.Series, live: float | None, today: dt.date) -> tuple[float, float]:
    """Most recent price + yesterday's close, handling intraday vs closed market."""
    last_bar = s.index[-1].date()
    if live is not None:
        current = live
        prev = float(s.iloc[-2]) if last_bar == today else float(s.iloc[-1])
    else:
        current = float(s.iloc[-1])
        prev = float(s.iloc[-2])
    return current, prev


def lookback(s: pd.Series, cutoff: pd.Timestamp) -> float:
    """Last close on or before the cutoff date (earliest available as fallback)."""
    past = s[s.index <= cutoff]
    return float(past.iloc[-1]) if len(past) else float(s.iloc[0])


def compute_table(close: pd.DataFrame, live: dict, today: dt.date) -> pd.DataFrame:
    now = pd.Timestamp(today)
    c3m, c1y, c5y = now - pd.DateOffset(months=3), now - pd.DateOffset(years=1), now - pd.DateOffset(years=5)
    rows = []
    for ticker, sector in SECTOR_ETFS.items():
        if ticker not in close.columns:
            continue
        s = close[ticker].dropna()
        if len(s) < 2:
            continue
        current, prev_close = current_and_prev(s, live.get(ticker), today)
        three_mo, one_yr, five_yr = lookback(s, c3m), lookback(s, c1y), lookback(s, c5y)
        rows.append({
            "Ticker": ticker, "Sector": sector, "Price": current,
            "Prev Close": prev_close,
            "Day %": (current - prev_close) / prev_close * 100,
            "3-Mo %": (current - three_mo) / three_mo * 100,
            "1-Yr %": (current - one_yr) / one_yr * 100,
            "5-Yr %": (current - five_yr) / five_yr * 100,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def style_row(row: pd.Series) -> list[str]:
    def c(v):
        return GREEN if v > 0 else RED if v < 0 else ""
    day = c(row["Day %"])
    return [day, day, "", "", day, c(row["3-Mo %"]), c(row["1-Yr %"]), c(row["5-Yr %"])]


def make_styler(df: pd.DataFrame):
    return df.style.apply(style_row, axis=1).format({
        "Price": "${:,.2f}", "Prev Close": "${:,.2f}",
        "Day %": "{:+.2f}%", "3-Mo %": "{:+.2f}%", "1-Yr %": "{:+.2f}%", "5-Yr %": "{:+.2f}%",
    })


# ═══════════════════════ Fed rate + money-flow summary ═══════════════════════

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def parse_fred_csv(text: str) -> tuple[float, str]:
    """Latest value + date from a FRED csv download (missing values are '.')."""
    df = pd.read_csv(io.StringIO(text))
    vals = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    idx = vals.last_valid_index()
    if idx is None:
        raise ValueError("no numeric data")
    return float(vals.loc[idx]), str(df.iloc[idx, 0])


def _fred_latest(series_id: str) -> tuple[float, str]:
    r = requests.get(FRED_CSV.format(sid=series_id), timeout=15)
    r.raise_for_status()
    return parse_fred_csv(r.text)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fed_rate() -> dict | None:
    """Fed funds target range from FRED (no API key), falling back to the
    effective rate. Cached for a day."""
    try:
        lo, _ = _fred_latest("DFEDTARL")
        hi, date = _fred_latest("DFEDTARU")
        return {"label": f"{lo:.2f}–{hi:.2f}%", "date": date, "kind": "target range"}
    except Exception:
        try:
            v, date = _fred_latest("DFF")
            return {"label": f"{v:.2f}%", "date": date, "kind": "effective"}
        except Exception:
            return None


def top_movers(df: pd.DataFrame, n: int = 3) -> tuple[list[str], list[str]]:
    """(top-N green sectors, top-N red sectors) by today's move."""
    greens = df[df["Day %"] > 0].nlargest(n, "Day %")["Sector"].tolist()
    reds = df[df["Day %"] < 0].nsmallest(n, "Day %")["Sector"].tolist()
    return greens, reds


# ═══════════════════════ PART B: deep scan (step 4) ═══════════════════════

def get_top_holdings(etf: str, n: int = 5) -> list[tuple[str, float | None]]:
    """Top-N holdings (symbol, weight-fraction) of an ETF via Yahoo fund data."""
    df = yf.Ticker(etf).funds_data.top_holdings
    if df is None or df.empty:
        raise ValueError("no holdings data returned")
    return parse_holdings_df(df, n)


def parse_holdings_df(df: pd.DataFrame, n: int) -> list[tuple[str, float | None]]:
    weight_col = next(
        (c for c in df.columns if "percent" in str(c).lower() or "%" in str(c)),
        None,
    )
    out = []
    for sym, row in df.head(n).iterrows():
        w = None
        if weight_col is not None:
            try:
                w = float(row[weight_col])
                if w > 1:  # given as 12.3 instead of 0.123
                    w /= 100
            except Exception:
                w = None
        out.append((str(sym).upper(), w))
    return out


def _is_auth_error(exc) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in AUTH_ERROR_MARKERS)


def _reset_yahoo_session() -> None:
    """Drop yfinance's cached cookie + crumb so the next request starts a fresh
    session — cures 401s caused by a stale session in a long-running app."""
    try:
        from yfinance.data import YfData
        d = YfData()
        if d._cookie_lock.acquire(timeout=5):
            try:
                d._crumb = None
                d._cookie = None
                sess = getattr(d, "_session", None)
                if sess is not None and not getattr(d, "_logged_in", False):
                    sess.cookies.clear()
            finally:
                d._cookie_lock.release()
    except Exception:
        pass


def load_holdings_cache(path: Path = HOLDINGS_CACHE_FILE) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_holdings_cache(cache: dict, path: Path = HOLDINGS_CACHE_FILE) -> None:
    try:
        path.write_text(json.dumps(cache, indent=1))
    except Exception:
        pass  # read-only folder: the app still works, just without the saved list


def cached_entry_if_fresh(entry, today: dt.date, max_age: int = HOLDINGS_CACHE_MAX_AGE_DAYS):
    """-> (holdings, date string) when the saved list is recent enough, else None."""
    try:
        saved = dt.date.fromisoformat(entry["date"])
        hold = [(str(s), None if w is None else float(w)) for s, w in entry["holdings"]]
        if hold and 0 <= (today - saved).days <= max_age:
            return hold, entry["date"]
    except Exception:
        pass
    return None


def _latest_shares(sym: str) -> float | None:
    """Most recent share count from Yahoo's share-count history."""
    try:
        s = yf.Ticker(sym).get_shares_full(start=dt.date.today() - dt.timedelta(days=400))
        if s is None or len(s) == 0:
            return None
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        v = float(s.dropna().iloc[-1])
        return v if v > 0 else None
    except Exception:
        return None


@st.cache_data(ttl=604800, show_spinner=False)
def candidate_shares(syms: tuple) -> dict:
    """Share counts barely move week to week — cached for 7 days."""
    out = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_latest_shares, s): s for s in syms}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


def market_caps_from(shares: dict, closes: pd.DataFrame) -> dict:
    caps = {}
    for sym, sh in shares.items():
        if sh and sym in closes.columns:
            px = closes[sym].dropna()
            if len(px):
                caps[sym] = float(sh) * float(px.iloc[-1])
    return caps


def rank_by_market_cap(candidates: list, caps: dict, n: int = 5) -> list | None:
    """Top-n by live market cap; None if too few caps came back to trust it."""
    ranked = sorted((c for c in candidates if caps.get(c)), key=lambda c: caps[c], reverse=True)
    return ranked[:n] if len(ranked) >= n else None


def resolve_holdings(etfs: tuple, today: dt.date, live_fetch=None, cache_path: Path = HOLDINGS_CACHE_FILE):
    """Live Yahoo holdings first; when blocked: recent saved list -> needs ranking.
    -> (holdings, sources, errors, need_rank). Stops asking Yahoo after it
    confirms the block, so a blocked scan stays fast and avoids rate limits."""
    live_fetch = live_fetch or get_top_holdings
    cache = load_holdings_cache(cache_path)
    holdings, sources, errors, need_rank = {}, {}, {}, []
    live_blocked, cache_dirty, last_err = False, False, {}
    for etf in etfs:
        if not live_blocked:
            got = None
            try:
                got = live_fetch(etf, 5)
            except Exception as exc:
                last_err[etf] = str(exc)
                if _is_auth_error(exc):
                    _reset_yahoo_session()
                    try:
                        got = live_fetch(etf, 5)
                    except Exception as exc2:
                        last_err[etf] = str(exc2)
                        if _is_auth_error(exc2):
                            live_blocked = True
            if got:
                holdings[etf] = got
                sources[etf] = ("live", today.isoformat())
                cache[etf] = {"date": today.isoformat(), "holdings": [list(h) for h in got]}
                cache_dirty = True
                continue
        fresh = cached_entry_if_fresh(cache.get(etf), today)
        if fresh:
            holdings[etf], sources[etf] = fresh[0], ("cache", fresh[1])
        elif etf in SECTOR_CANDIDATES:
            need_rank.append(etf)
        else:
            errors[etf] = last_err.get(etf, "Yahoo fund data blocked")
    if cache_dirty:
        save_holdings_cache(cache, cache_path)
    return holdings, sources, errors, need_rank


def macd_state(close: pd.Series) -> dict | None:
    """Classic MACD 12/26/9. Bullish = MACD line above its signal line."""
    if len(close) < 35:
        return None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return {
        "bullish": bool(macd.iloc[-1] > signal.iloc[-1]),
        "rising": bool(hist.iloc[-1] > hist.iloc[-2]),
    }


def fib_state(close: pd.Series, price: float) -> dict | None:
    """Where price sits in the 6-month range, vs Fibonacci retracement levels."""
    if len(close) < 30:
        return None
    hi, lo = float(close.max()), float(close.min())
    if hi <= lo:
        return None
    pct = (price - lo) / (hi - lo)
    zone = "strong" if pct >= 0.618 else "weak" if pct <= 0.382 else "neutral"
    for level in (0.236, 0.382, 0.5, 0.618, 0.786):
        if pct <= level:
            nearest = level
            break
    else:
        nearest = 1.0
    return {"pct": pct, "zone": zone, "nearest": nearest, "hi": hi, "lo": lo}


def parse_news_items(raw: list, limit: int = 2) -> list[str]:
    """Titles from yfinance news (old and new payload shapes), tagged if political."""
    heads = []
    for item in raw or []:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or ""
        if not title:
            continue
        low = title.lower()
        tag = "🏛️ " if any(k in low for k in POLITICAL_KEYWORDS) else ""
        if "earning" in low:
            tag += "📅 "
        heads.append(tag + title)
        if len(heads) >= limit:
            break
    return heads


# ═══════════════════════ SEC EDGAR: financials straight from the filings ═══════════════════════
# Companies file their 10-K (annual) and 10-Q (quarterly) reports with the SEC in
# machine-readable XBRL, and data.sec.gov serves those exact numbers for free —
# the original source Yahoo and every data vendor copy from. SEC's only rules:
# each app identifies itself with a name + email (sent only to sec.gov, in the
# request header) and stays under 10 requests per second.

SEC_DATA = "https://data.sec.gov"
SEC_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_CACHE_DIR = Path(__file__).with_name("sec_cache")
SEC_CONTACT_FILE = Path(__file__).with_name("sec_contact.txt")
SEC_PARSER_VERSION = 1        # bump to invalidate saved SEC data after parser changes
SEC_MAX_RPS = 8               # SEC's limit is 10 requests/sec — stay safely under it
SEC_HISTORY_YEARS = 10        # years of 10-K history the Compounder scores on
PRICE_NORM_YEARS = 5          # EV/FCF averages this many recent years of real FCF
SEC_RESTATE_PCT = 0.10        # a past year's revenue re-filed >10% different = new reporting basis

# Ticker -> SEC company ID (CIK) for every built-in candidate, checked against
# EDGAR in Oct 2026. Any other ticker is looked up in SEC's own ticker file.
SEC_CIKS = {
    "NVDA": 1045810, "AAPL": 320193, "MSFT": 789019, "AVGO": 1730168, "ORCL": 1341439,
    "MU": 723125, "AMD": 2488, "PLTR": 1321655, "CSCO": 858877, "IBM": 51143,
    "CRM": 1108524, "AMAT": 6951, "LRCX": 707549, "INTU": 896878,
    "META": 1326801, "GOOGL": 1652044, "GOOG": 1652044, "NFLX": 1065280, "TMUS": 1283699,
    "DIS": 1744489, "T": 732717, "VZ": 732712, "CMCSA": 1166691, "TTWO": 946581,
    "CHTR": 1091667,
    "AMZN": 1018724, "TSLA": 1318605, "HD": 354950, "MCD": 63908, "BKNG": 1075531,
    "TJX": 109198, "LOW": 60667, "SBUX": 829224, "NKE": 320187, "ABNB": 1559720,
    "ORLY": 898173, "RCL": 884887, "MAR": 1048286, "CMG": 1058090,
    "WMT": 104169, "COST": 909832, "PG": 80424, "KO": 21344, "PEP": 77476, "PM": 1413329,
    "MO": 764180, "MDLZ": 1103982, "CL": 21665, "MNST": 865752, "TGT": 27419, "KDP": 1418135,
    "XOM": 2115436, "CVX": 93410, "COP": 1163165, "WMB": 107263, "EOG": 821189,
    "KMI": 1506307, "OKE": 1039684, "MPC": 1510295, "PSX": 1534701, "SLB": 87347,
    "VLO": 1035002, "OXY": 797468, "BKR": 1701605,
    "BRK-B": 1067983, "JPM": 19617, "V": 1403161, "MA": 1141391, "BAC": 70858,
    "WFC": 72971, "GS": 886982, "MS": 895421, "AXP": 4962, "C": 831001, "SCHW": 316709,
    "BLK": 2012383, "SPGI": 64040, "PGR": 80661,
    "LLY": 59478, "JNJ": 200406, "ABBV": 1551152, "UNH": 731766, "MRK": 310158,
    "ABT": 1800, "TMO": 97745, "ISRG": 1035267, "AMGN": 318154, "PFE": 78003,
    "BSX": 885725, "DHR": 313616, "GILD": 882095, "SYK": 310764, "VRTX": 875320,
    "GE": 40545, "CAT": 18230, "RTX": 101829, "GEV": 1996810, "UBER": 1543151,
    "HON": 773840, "BA": 12927, "UNP": 100885, "ETN": 1551182, "DE": 315189,
    "LMT": 936468, "ADP": 8670, "PH": 76334, "TT": 1466258,
    "LIN": 1707925, "SHW": 89800, "NEM": 1164727, "ECL": 31462, "APD": 2969,
    "FCX": 831259, "CTVA": 1755672, "NUE": 73309, "DD": 1666700, "VMC": 1396009,
    "MLM": 916076, "DOW": 1751788,
    "PLD": 1045609, "AMT": 1053507, "WELL": 766704, "EQIX": 1101239, "SPG": 1063761,
    "O": 726728, "PSA": 1393311, "DLR": 1297996, "CCI": 1051470, "CBRE": 1138118,
    "VICI": 1705696, "EXR": 1289490,
    "NEE": 753308, "SO": 92122, "CEG": 1868275, "DUK": 1326160, "VST": 1692819,
    "AEP": 4904, "SRE": 1032208, "D": 715957, "EXC": 1109357, "PCG": 1004980,
    "XEL": 72903, "PEG": 788784, "ED": 1047862,
}
# A company re-organized under a new CIK keeps its older filings under the old one
SEC_PREDECESSORS = {
    2115436: [34088],     # ExxonMobil — July 2026 holding-company reorganization
    2012383: [1364742],   # BlackRock — 2024 holding-company reorganization (GIP deal)
    1744489: [1001039],   # Disney — 2019 holding company (21st Century Fox deal)
}

# XBRL tags, in priority order. Companies pick different tags for the same line
# (and switch over the years), so each line tries the alternatives.
SEC_REVENUE = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
               "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense",
               "RegulatedAndUnregulatedOperatingRevenue", "SalesRevenueNet"]
SEC_NETINC = ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"]
SEC_OPINC = ["OperatingIncomeLoss"]
SEC_PRETAX = ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
              "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]
SEC_PRETAX_PARTS = ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
                    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign"]
SEC_INTEREST = ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"]
SEC_SHARES = ["WeightedAverageNumberOfDilutedSharesOutstanding",
              "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
              "WeightedAverageNumberOfSharesOutstandingBasic"]
SEC_EQUITY = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
SEC_CASH = ["CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash", "CashAndDueFromBanks"]
SEC_STI = ["MarketableSecuritiesCurrent", "ShortTermInvestments", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"]
SEC_ST_DEBT = ["ShortTermBorrowings", "CommercialPaper", "OtherShortTermBorrowings"]
SEC_LT_DEBT_ALL = ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"]
SEC_LT_DEBT_NONCUR = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"]
SEC_LT_DEBT_CUR = ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"]
SEC_DEBT_TOTAL = ["DebtLongtermAndShorttermCombinedAmount", "DebtInstrumentCarryingAmount"]
SEC_DEBT_PIECES = ["SeniorNotes", "SecuredDebt", "UnsecuredDebt", "NotesPayable", "LineOfCredit"]
SEC_OCF = ["NetCashProvidedByUsedInOperatingActivities",
           "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
SEC_CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
             "PaymentsForCapitalImprovements", "PaymentsToAcquireOtherPropertyPlantAndEquipment"]
SEC_SBC = ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"]
SEC_DEFREV = ["ContractWithCustomerLiability", "DeferredRevenue"]
SEC_DEFREV_PARTS = [("ContractWithCustomerLiabilityCurrent", "ContractWithCustomerLiabilityNoncurrent"),
                    ("DeferredRevenueCurrent", "DeferredRevenueNoncurrent")]
SEC_DEI_SHARES = "EntityCommonStockSharesOutstanding"
SEC_WANTED = set(SEC_REVENUE + SEC_NETINC + SEC_OPINC + SEC_PRETAX + SEC_PRETAX_PARTS + SEC_INTEREST
                 + SEC_SHARES + SEC_EQUITY + SEC_CASH + SEC_STI + SEC_ST_DEBT + SEC_LT_DEBT_ALL
                 + SEC_LT_DEBT_NONCUR + SEC_LT_DEBT_CUR + SEC_DEBT_TOTAL + SEC_DEBT_PIECES + SEC_OCF + SEC_CAPEX + SEC_SBC + SEC_DEFREV
                 + [t for pair in SEC_DEFREV_PARTS for t in pair]
                 + ["DebtCurrent", "CashCashEquivalentsAndShortTermInvestments", SEC_DEI_SHARES])
SEC_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A"}

# 8-K "items" — the reason a company filed a current report
SEC_8K_ITEMS = {
    "1.01": "major agreement", "1.02": "agreement ended", "1.03": "bankruptcy",
    "1.05": "cybersecurity incident", "2.01": "acquisition/sale closed", "2.02": "earnings release",
    "2.03": "new debt", "2.04": "debt default/acceleration", "2.05": "restructuring/layoffs",
    "2.06": "asset write-down", "3.01": "listing notice", "3.02": "unregistered share sale",
    "3.03": "shareholder-rights change", "4.01": "auditor change",
    "4.02": "restatement — past financials can't be relied on", "5.01": "change in control",
    "5.02": "exec/board change", "5.03": "bylaw change", "5.07": "shareholder vote",
    "7.01": "investor update", "8.01": "other news",
}
SEC_RED_FLAGS = {"1.03", "2.04", "4.02"}                      # block a green dot for a year
SEC_WATCH = {"1.05", "2.05", "2.06", "3.01", "4.01", "5.01"}  # shown with ⚠, don't block
SEC_EVENT_DAYS, SEC_WATCH_DAYS, SEC_FLAG_DAYS = 45, 180, 365
HEALTH_KEYS = ("revenueGrowth", "profitMargins", "returnOnEquity", "debtToEquity", "freeCashflow")
SEC_PENDING_RETRY_HOURS = 12   # data incomplete (SEC still processing a filing): re-check this often


def sec_contact() -> str:
    """The name + email SEC asks every app to send: a Streamlit secret, an
    environment variable, or the one saved from the in-app box."""
    v = ""
    try:
        v = str(st.secrets.get("SEC_CONTACT", "") or "")
    except Exception:
        pass
    v = v or os.environ.get("SEC_CONTACT", "")
    if not v:
        try:
            v = SEC_CONTACT_FILE.read_text(encoding="utf-8")
        except OSError:
            v = ""
    return v.strip()


def valid_contact(v: str) -> bool:
    return bool(re.search(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", v or ""))


def save_contact(v: str) -> bool:
    try:
        SEC_CONTACT_FILE.write_text(v.strip(), encoding="utf-8")
        return True
    except OSError:
        return False


class SecBlocked(Exception):
    pass


class SecClient:
    """Polite EDGAR client: identifies itself and paces requests under SEC's limit
    (shared by all worker threads)."""

    def __init__(self, contact: str):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"FidelitySectorScanner/2.0 {contact}",
                                     "Accept-Encoding": "gzip, deflate"})
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def get_json(self, url: str):
        return self.get(url).json()

    def get(self, url: str) -> requests.Response:
        resp = None
        for attempt in range(3):
            with self._lock:
                now = time.monotonic()
                wait = self._next_slot - now
                self._next_slot = max(now, self._next_slot) + 1.0 / SEC_MAX_RPS
            if wait > 0:
                time.sleep(wait)
            resp = self.session.get(url, timeout=45)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1))
                continue
            if resp.status_code == 403:
                raise SecBlocked("SEC refused the request (403) — check the name/email, "
                                 "or SEC is throttling; try again in a minute")
            if resp.status_code == 404:
                raise FileNotFoundError("no data at SEC for this company")
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"SEC busy (HTTP {resp.status_code if resp is not None else '?'})")


_SEC_CLIENTS: dict = {}
_SEC_LOCK = threading.Lock()
_SEC_FACTS_GATE = threading.Semaphore(3)   # big downloads: at most 3 parsed at once (memory)
_SEC_CIK_LOCKS: dict = {}


def sec_client(contact: str) -> SecClient:
    with _SEC_LOCK:
        if contact not in _SEC_CLIENTS:
            _SEC_CLIENTS[contact] = SecClient(contact)
        return _SEC_CLIENTS[contact]


def _cik_lock(cik: int) -> threading.Lock:
    with _SEC_LOCK:
        return _SEC_CIK_LOCKS.setdefault(cik, threading.Lock())


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj) -> None:
    try:
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass   # read-only disk: still works, just re-downloads next time


def sec_ticker_map(client: SecClient) -> dict:
    """Ticker -> CIK for every SEC filer, from SEC's own list (saved for a week)."""
    path = SEC_CACHE_DIR / "company_tickers.json"
    with _SEC_LOCK:
        try:
            if path.exists() and time.time() - path.stat().st_mtime < 7 * 86400:
                cached = _read_json(path)
                if cached:
                    return cached
        except OSError:
            pass
    raw = client.get_json(SEC_TICKERS_URL)
    m = {str(v["ticker"]).upper().replace(".", "-"): int(v["cik_str"]) for v in raw.values()}
    _write_json(path, m)
    return m


def cik_for(sym: str, client: SecClient) -> int | None:
    key = sym.upper().replace(".", "-")
    if key in SEC_CIKS:
        return SEC_CIKS[key]
    try:
        return sec_ticker_map(client).get(key)
    except Exception:
        return None


def _days(a: str, b: str) -> int:
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def _first(g, names):
    for n in names:
        v = g(n)
        if v is not None:
            return v
    return None


def _largest(g, names):
    """Revenue: a company can tag a total AND a narrower subtotal — take the total."""
    vals = [v for v in (g(n) for n in names) if v is not None]
    return max(vals) if vals else None


class SecFacts:
    """One company's XBRL facts indexed by period. When a later filing re-reports
    a period (restatement, re-cast), the most recent filing's number wins."""

    def __init__(self, payloads: list | None = None):
        self.hist: dict = {}    # tag -> {(start, end): [(filed, val, form), ...]}
        self.inst: dict = {}    # tag -> {end: (filed, val)}
        self.accns: set = set()
        for p in payloads or []:
            self.add(p)
        self.finish()

    def add(self, payload: dict) -> None:
        """Add one company-facts JSON (or one parsed filing); call finish() after."""
        facts = (payload or {}).get("facts", {})
        for ns in ("us-gaap", "dei"):
            for tag, body in facts.get(ns, {}).items():
                if tag not in SEC_WANTED:
                    continue
                for unit, rows in body.get("units", {}).items():
                    if unit not in ("USD", "shares"):
                        continue
                    for x in rows:
                        if x.get("form") not in SEC_FORMS or "end" not in x:
                            continue
                        self.accns.add(x.get("accn"))
                        filed, val, end = x.get("filed", ""), float(x["val"]), x["end"]
                        if "start" in x:
                            self.hist.setdefault(tag, {}).setdefault((x["start"], end), []).append(
                                (filed, val, x.get("form", "")))
                        else:
                            d = self.inst.setdefault(tag, {})
                            if end not in d or filed >= d[end][0]:
                                d[end] = (filed, val)

    def finish(self) -> None:
        self.dur = {t: {k: max(rows, key=lambda r: r[0])[:2] for k, rows in per.items()}
                    for t, per in self.hist.items()}
        self._annual: dict = {}

    def annual(self, tag: str) -> dict:
        """{fiscal-year end: (filed, value)} for fiscal years reported in a 10-K
        (12-month figures that only appear in a 10-Q are trailing-twelve-month
        extras, e.g. Amazon's free-cash-flow table — not fiscal years)."""
        if tag not in self._annual:
            out = {}
            for (s, e), (filed, v) in self.dur.get(tag, {}).items():
                if 340 <= _days(s, e) <= 390 and (e not in out or filed > out[e][0]) \
                        and any(r[2].startswith("10-K") for r in self.hist[tag][(s, e)]):
                    out[e] = (filed, v)
            self._annual[tag] = out
        return self._annual[tag]

    def val(self, tag: str, start: str, end: str):
        r = self.dur.get(tag, {}).get((start, end))
        return r[1] if r else None

    def at(self, tag: str, end: str):
        """Balance-sheet value on a date (±7 days for 52/53-week calendars)."""
        d = self.inst.get(tag, {})
        if end in d:
            return d[end][1]
        near = [(abs(_days(e, end)), e) for e in d if abs(_days(e, end)) <= 7]
        return d[min(near)[1]][1] if near else None

    def latest_inst(self, tag: str):
        d = self.inst.get(tag, {})
        return d[max(d)][1] if d else None

    def quarter(self, tag: str, end: str):
        """One fiscal quarter's value. Q4 isn't filed on its own — it's the
        full year minus the first nine months."""
        per = self.dur.get(tag, {})
        for (s, e), (_, v) in per.items():
            if e == end and 80 <= _days(s, e) <= 100:
                return v
        for (s, e), (_, v) in per.items():
            if e == end and 340 <= _days(s, e) <= 390:
                nine = [(e2, v2) for (s2, e2), (_, v2) in per.items()
                        if s2 == s and 250 <= _days(s2, e2) <= 290]
                if nine:
                    return v - max(nine)[1]
        return None

    def restated_since(self) -> str | None:
        """Filing date of the newest big re-cast of a past year's revenue (spin-off,
        divestiture, reporting change). Years only reported before it sit on the
        old basis and would distort 10-year trends."""
        newest = None
        for tag in SEC_REVENUE:
            for (s, e), rows in self.hist.get(tag, {}).items():
                if len(rows) < 2 or not 340 <= _days(s, e) <= 390:
                    continue
                rows = sorted(rows)
                first = rows[0][1]
                if first <= 0:
                    continue
                for filed, v, _ in rows[1:]:
                    if abs(v / first - 1) > SEC_RESTATE_PCT:
                        newest = max(newest or filed, filed)
                        break
        return newest

    def split_events(self) -> list:
        """Stock splits, found where a later filing re-reports an old period's share
        count by a split-like ratio (2:1, 3:2, 10:1, 1:4 …). -> [(date, ratio)]."""
        found = []
        for tag in SEC_SHARES:
            for rows in self.hist.get(tag, {}).values():
                rows = sorted(rows)
                for (f0, v0, _), (f1, v1, _) in zip(rows, rows[1:]):
                    if v0 <= 0 or v1 <= 0:
                        continue
                    r = v1 / v0
                    big = r if r >= 1 else 1 / r
                    # real splits run 3:2 … 50:1; ×1000-type jumps are tagging mistakes
                    if 1.4 <= big <= 60 and abs(big - round(big * 2) / 2) / big < 0.02:
                        found.append((f1, r))
        merged = []
        for f, r in sorted(found):
            if any(abs(r / r2 - 1) < 0.05 and abs(_days(f2, f)) < 400 for f2, r2 in merged):
                continue
            merged.append((f, r))
        return merged

    def periods(self, today: str) -> dict | None:
        """The newest reported period: its year-to-date span, the same span a year
        earlier, and the fiscal year that ended just before it."""
        best = None
        for tag in SEC_REVENUE + SEC_NETINC:
            fy = self.annual(tag)
            for (s, e) in self.dur.get(tag, {}):
                d = _days(s, e)
                if e > today or d < 80 or (d > 290 and e not in fy):
                    continue
                if best is None or (e, d) > best[0]:
                    best = ((e, d), tag, s, e, d)
        if not best:
            return None
        _, tag, S, E, D = best
        annual = D >= 340
        prior = None
        for (s2, e2) in self.dur[tag]:
            if abs(_days(e2, E) - 365) <= 10 and abs(_days(s2, e2) - D) <= 10:
                prior = (s2, e2)
        fy_end = None
        if not annual:
            for e2 in self.annual(tag):
                if abs(_days(e2, S) - 1) <= 10:
                    fy_end = e2
        if not annual and (prior is None or fy_end is None):
            # can't build trailing-12-months: fall back to the last full year
            ends = sorted(self.annual(tag))
            if not ends:
                return None
            E = ends[-1]
            S = next(s for (s, e) in self.dur[tag] if e == E and 340 <= _days(s, e) <= 390)
            annual, fy_end = True, None
            prior = next(((s2, e2) for (s2, e2) in self.dur[tag]
                          if abs(_days(e2, E) - 365) <= 10 and 340 <= _days(s2, e2) <= 390), None)
        return {"start": S, "end": E, "annual": annual, "prior": prior, "fy_end": fy_end}

    def ttm(self, tag: str, P: dict):
        """Trailing 12 months: last full year + this year-to-date − last year's same span."""
        if P["annual"]:
            r = self.annual(tag).get(P["end"])
            return r[1] if r else None
        fy = self.annual(tag).get(P["fy_end"])
        cur = self.val(tag, P["start"], P["end"])
        pri = self.val(tag, *P["prior"])
        if fy is None or cur is None or pri is None:
            return None
        return fy[1] + cur - pri


def _pretax(g):
    v = _first(g, SEC_PRETAX)
    if v is None:
        parts = [g(t) for t in SEC_PRETAX_PARTS]
        if all(p is not None for p in parts):
            v = sum(parts)
    return v


def _ebit(g, financial: bool):
    """Operating income; companies that don't report one (drugmakers, banks,
    conglomerates) get pre-tax income + interest expense (banks: pre-tax income,
    since interest is their cost of doing business)."""
    v = g(SEC_OPINC[0])
    if v is not None:
        return v
    p = _pretax(g)
    if p is None or financial:
        return p
    return p + abs(_first(g, SEC_INTEREST) or 0.0)


def _debt(g):
    st_ = _first(g, SEC_ST_DEBT)
    total = _first(g, SEC_LT_DEBT_ALL)            # long-term debt incl. current portion
    if total is not None:
        return total + (st_ or 0.0)
    noncur, cur = _first(g, SEC_LT_DEBT_NONCUR), g("DebtCurrent")
    if noncur is not None and cur is not None:    # DebtCurrent already includes short-term
        return noncur + cur
    if noncur is not None:
        return noncur + (_first(g, SEC_LT_DEBT_CUR) or 0.0) + (st_ or 0.0)
    v = _first(g, SEC_DEBT_TOTAL)                 # footnote total (insurers, some REITs)
    if v is not None:
        return v
    pieces = [x for x in (g(t) for t in SEC_DEBT_PIECES) if x is not None]
    if pieces:
        return sum(pieces)
    if cur is not None or st_ is not None:
        return cur if cur is not None else st_
    return None


def _cash(g):
    v = g("CashCashEquivalentsAndShortTermInvestments")
    if v is not None:
        return v
    c = _first(g, SEC_CASH)
    if c is None:
        return None
    return c + (_first(g, SEC_STI) or 0.0)     # cash + short-term marketable securities


def _fcf(g):
    ocf, capex = _first(g, SEC_OCF), _first(g, SEC_CAPEX)
    if ocf is None or capex is None:
        return None     # capex under a company-specific tag: leave it to Yahoo
    return ocf - abs(capex)


def _defrev(g):
    v = _first(g, SEC_DEFREV)
    if v is not None:
        return v
    for cur, non in SEC_DEFREV_PARTS:
        c = g(cur)
        if c is not None:
            return c + (g(non) or 0.0)
    return None


def sec_extract(fx: SecFacts, financial: bool, today: str) -> dict:
    """Everything the scanner scores, from one company's facts (JSON-safe)."""
    # ---- annual history (Compounder) ----
    ends = sorted({e for t in SEC_REVENUE + SEC_NETINC for e in fx.annual(t) if e <= today},
                  reverse=True)[:SEC_HISTORY_YEARS]
    cut = fx.restated_since()
    basis_note = None
    if cut:
        def filed_for(e):
            return max((fx.annual(t)[e][0] for t in SEC_REVENUE + SEC_NETINC if e in fx.annual(t)),
                       default="")
        kept = [e for e in ends if filed_for(e) >= cut]
        if 2 <= len(kept) < len(ends):
            ends = kept
            basis_note = "earlier years re-cast after a spin-off/divestiture"
    events = fx.split_events()
    sh_annual = {}
    for t in SEC_SHARES:
        for e, (filed, v) in fx.annual(t).items():
            if e not in sh_annual:
                factor = 1.0
                for d, r in events:
                    if d > filed:
                        factor *= r
                sh_annual[e] = v * factor
    if sh_annual:   # drop tagging slips (years filed in thousands by mistake), judged
        anchor = sh_annual[max(sh_annual)]          # against the latest 10-K's count
        sh_annual = {e: v for e, v in sh_annual.items() if anchor / 20 <= v <= anchor * 20}
    lines = {k: {} for k in ("revenue", "op_income", "ebit", "net_income", "shares", "equity",
                             "debt", "cash", "deferred_rev", "fcf", "ocf", "sbc")}
    for e in ends:
        gf = lambda t, e=e: (fx.annual(t).get(e) or (None, None))[1]
        gi = lambda t, e=e: fx.at(t, e)
        vals = {
            "revenue": _largest(gf, SEC_REVENUE), "op_income": gf(SEC_OPINC[0]),
            "ebit": _ebit(gf, financial), "net_income": _first(gf, SEC_NETINC),
            "shares": sh_annual.get(e), "equity": _first(gi, SEC_EQUITY), "debt": _debt(gi),
            "cash": _cash(gi), "deferred_rev": _defrev(gi), "fcf": _fcf(gf),
            "ocf": _first(gf, SEC_OCF), "sbc": _first(gf, SEC_SBC),
        }
        for k, v in vals.items():
            if v is not None:
                lines[k][e] = v

    # ---- latest quarter / trailing 12 months (Dip-buyer health, EV) ----
    latest = {}
    P = fx.periods(today)
    if P:
        E = P["end"]
        g = lambda t: fx.ttm(t, P)
        gi = lambda t: fx.at(t, E)
        rev, ni, fcf = _largest(g, SEC_REVENUE), _first(g, SEC_NETINC), _fcf(g)
        eq, debt, cash = _first(gi, SEC_EQUITY), _debt(gi), _cash(gi)
        prior_end = P["prior"][1] if P["prior"] else None
        q_now = _largest(lambda t: fx.quarter(t, E), SEC_REVENUE)
        q_then = _largest(lambda t: fx.quarter(t, prior_end), SEC_REVENUE) if prior_end else None
        latest = {
            "end": E, "ttm": not P["annual"],
            "revenueGrowth": (q_now / q_then - 1) if q_now and q_then and q_then > 0 else None,
            "profitMargins": (ni / rev) if ni is not None and rev and rev > 0 else None,
            "returnOnEquity": (ni / eq) if ni is not None and eq and eq > 0 else None,
            "debtToEquity": (debt / eq * 100) if debt is not None and eq and eq > 0 else None,
            "freeCashflow": fcf, "debt": debt, "cash": cash, "equity": eq,
        }
        if fcf is None and lines["fcf"]:          # capex not in the 10-Q: last full year
            latest["freeCashflow"] = lines["fcf"][max(lines["fcf"])]
    # shares outstanding now: the cover-page count, but only when it agrees with the
    # diluted count — multi-class companies (Berkshire, Visa) list just one class there
    sh_now = fx.latest_inst(SEC_DEI_SHARES)
    sh_last = sh_annual.get(ends[0]) if ends else None
    if sh_now is None or not sh_last or not 0.75 <= sh_now / sh_last <= 1.25:
        sh_now = sh_last
    return {"lines": lines, "latest": latest, "shares_now": sh_now, "basis_note": basis_note,
            "splits": events, "years": len(ends)}


def sec_filings(sub: dict, cik: int, today: dt.date) -> dict:
    """Latest 10-K / 10-Q (with links) and recent 8-K events from the filing index."""
    r = (sub or {}).get("filings", {}).get("recent", {})
    col = lambda name, i: (r.get(name) or [""] * (i + 1))[i] or ""
    latest, events, watch, flags = {}, [], [], []
    newest, seen = None, set()
    for i in range(len(r.get("form", []))):
        form, fdate, accn = col("form", i), col("filingDate", i), col("accessionNumber", i)
        doc, rdate, items = col("primaryDocument", i), col("reportDate", i), col("items", i)
        url = SEC_DOC_URL.format(cik=cik, accn=accn.replace("-", ""), doc=doc) if doc else \
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
        try:
            age = (today - dt.date.fromisoformat(fdate)).days
        except ValueError:
            continue
        base = form.split("/")[0]
        if base in ("10-K", "10-Q"):
            f = {"form": form, "filed": fdate, "period": rdate, "url": url, "accn": accn, "doc": doc}
            if newest is None or (fdate, accn) > (newest["filed"], newest["accn"]):
                newest = f
            if base not in latest or fdate > latest[base]["filed"]:
                latest[base] = f
        if form.startswith("NT 10-") and age <= SEC_FLAG_DAYS:
            flags.append({"label": f"late-filing notice ({form})", "date": fdate, "url": url})
        if base == "8-K":
            for it in (x.strip() for x in items.split(",")):
                label = SEC_8K_ITEMS.get(it)
                if not label:
                    continue
                ev = {"item": it, "label": label, "date": fdate, "url": url}
                if it in SEC_RED_FLAGS and age <= SEC_FLAG_DAYS:
                    flags.append(ev)
                elif it in SEC_WATCH and age <= SEC_WATCH_DAYS:
                    watch.append(ev)
                elif age <= SEC_EVENT_DAYS and label not in seen:
                    seen.add(label)
                    events.append(ev)
    by_date = lambda lst: sorted(lst, key=lambda e: e["date"], reverse=True)
    return {"latest": latest, "newest": newest, "key": newest["accn"] if newest else "",
            "events": by_date(events), "watch": by_date(watch), "flags": by_date(flags)}


XBRLI = "{http://www.xbrl.org/2003/instance}"
XSI_NIL = "{http://www.w3.org/2001/XMLSchema-instance}nil"


def parse_xbrl_instance(xml_bytes: bytes, accn: str, form: str, filed: str) -> dict:
    """A filing's own XBRL instance -> the same shape as SEC's company-facts JSON.
    Company-wide numbers only: segment/dimension breakdowns are skipped, as SEC does."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_bytes)
    contexts, units = {}, {}
    for ctx in root.iter(XBRLI + "context"):
        if ctx.find(f".//{XBRLI}segment") is not None or ctx.find(f".//{XBRLI}scenario") is not None:
            continue
        per = ctx.find(XBRLI + "period")
        if per is None:
            continue
        inst = (per.findtext(XBRLI + "instant") or "").strip()
        s, e = (per.findtext(XBRLI + "startDate") or "").strip(), (per.findtext(XBRLI + "endDate") or "").strip()
        if inst:
            contexts[ctx.get("id")] = {"end": inst[:10]}
        elif s and e:
            contexts[ctx.get("id")] = {"start": s[:10], "end": e[:10]}
    for u in root.iter(XBRLI + "unit"):
        m = (u.findtext(XBRLI + "measure") or "").strip().lower()
        if m in ("iso4217:usd", "usd"):
            units[u.get("id")] = "USD"
        elif m.split(":")[-1] == "shares":
            units[u.get("id")] = "shares"
    facts = {"us-gaap": {}, "dei": {}}
    for el in root:
        if not isinstance(el.tag, str) or not el.tag.startswith("{"):
            continue
        uri, name = el.tag[1:].split("}", 1)
        ns = "us-gaap" if "fasb.org/us-gaap" in uri else "dei" if "xbrl.sec.gov/dei" in uri else None
        if ns is None or name not in SEC_WANTED or el.get(XSI_NIL) == "true":
            continue
        ctx, unit = contexts.get(el.get("contextRef")), units.get(el.get("unitRef"))
        if ctx is None or unit is None:
            continue
        try:
            val = float((el.text or "").strip())
        except ValueError:
            continue
        row = dict(ctx, val=val, accn=accn, form=form, filed=filed)
        facts[ns].setdefault(name, {"units": {}})["units"].setdefault(unit, []).append(row)
    return {"facts": facts}


def sec_filing_xbrl(client: SecClient, cik: int, filing: dict) -> dict | None:
    """Read one 10-K/10-Q's own XBRL straight from the EDGAR archive — used when
    SEC's data API hasn't added that filing yet (it can lag weeks for some companies)."""
    folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{filing['accn'].replace('-', '')}/"
    names = []
    doc = filing.get("doc") or ""
    if doc.lower().endswith((".htm", ".html")):
        names.append(doc.rsplit(".", 1)[0] + "_htm.xml")      # EDGAR's extracted instance
    data = None
    for name in names:
        try:
            data = client.get(folder + name).content
            break
        except FileNotFoundError:
            continue
    if data is None:
        idx = client.get_json(folder + "index.json")
        items = [it.get("name", "") for it in idx.get("directory", {}).get("item", [])]
        cands = [n for n in items if n.endswith("_htm.xml")] or \
                [n for n in items if n.endswith(".xml") and n != "FilingSummary.xml"
                 and not n.endswith(("_cal.xml", "_def.xml", "_lab.xml", "_pre.xml"))]
        if not cands:
            return None
        data = client.get(folder + cands[0]).content
    parsed = parse_xbrl_instance(data, filing["accn"], filing["form"], filing["filed"])
    usgaap = parsed["facts"]["us-gaap"]
    return parsed if any(t in usgaap for t in SEC_REVENUE + SEC_NETINC) else None


def _lines_to_series(lines: dict) -> dict:
    return {k: (pd.Series({pd.Timestamp(e): float(v) for e, v in d.items()}).sort_index(ascending=False)
                if d else None) for k, d in lines.items()}


def sec_company(sym: str, contact: str) -> dict:
    """SEC fundamentals + filings for one ticker. Never raises. The big company-facts
    download is saved in sec_cache/ and only refreshed after a new 10-K/10-Q."""
    try:
        client = sec_client(contact)
        cik = cik_for(sym, client)
        if not cik:
            return {"ok": False, "error": "no SEC filer for this ticker"}
        today = dt.date.today()
        sub = client.get_json(f"{SEC_DATA}/submissions/CIK{cik:010d}.json")
        filings = sec_filings(sub, cik, today)
        with _cik_lock(cik):
            path = SEC_CACHE_DIR / f"{cik}.json"
            cached = _read_json(path) or {}
            reuse = (cached.get("v") == SEC_PARSER_VERSION and cached.get("for") == filings["key"]
                     and (cached.get("complete") or
                          time.time() - cached.get("saved", 0) < SEC_PENDING_RETRY_HOURS * 3600))
            if reuse:
                data = cached["data"]
            else:
                stale = None
                with _SEC_FACTS_GATE:
                    fx = SecFacts()
                    for c in [cik] + SEC_PREDECESSORS.get(cik, []):
                        fx.add(client.get_json(f"{SEC_DATA}/api/xbrl/companyfacts/CIK{c:010d}.json"))
                    newest = filings["newest"]
                    if newest and newest["accn"] not in fx.accns:
                        try:
                            extra = sec_filing_xbrl(client, cik, newest)
                        except Exception:
                            extra = None
                        if extra:
                            fx.add(extra)
                        else:
                            stale = {k: newest[k] for k in ("form", "filed", "period")}
                    fx.finish()
                sic = str(sub.get("sic") or "")
                data = sec_extract(fx, financial=sic.isdigit() and 6000 <= int(sic) <= 6799,
                                   today=today.isoformat())
                data["stale"] = stale
                _write_json(path, {"v": SEC_PARSER_VERSION, "for": filings["key"], "complete": stale is None,
                                   "saved": time.time(), "data": data})
        return {"ok": True, "cik": cik, "name": sub.get("name") or "", "filings": filings,
                "annual": _lines_to_series(data["lines"]), "latest": data["latest"],
                "shares_now": data["shares_now"], "basis_note": data["basis_note"],
                "years": data["years"], "splits": data["splits"], "stale": data.get("stale")}
    except Exception as exc:
        return {"ok": False, "error": str(exc) or type(exc).__name__}


SEC_STMT_LINES = ("revenue", "op_income", "ebit", "net_income", "shares", "equity", "debt",
                  "cash", "deferred_rev", "fcf", "ocf", "sbc")
SEC_CORE_LINES = ("revenue", "ebit", "net_income", "shares", "equity", "debt", "cash", "fcf")


def sec_needs_yahoo(sec: dict) -> bool:
    """Does the Compounder need Yahoo's statements to fill gaps in the SEC data?"""
    if not sec or not sec.get("ok"):
        return True
    a = sec.get("annual") or {}
    return any(a.get(k) is None or len(a[k]) < 2 for k in SEC_CORE_LINES)


def merge_stmts(sec: dict, yahoo: dict) -> tuple[dict, list[str]]:
    """Per line: SEC filings first; Yahoo only where SEC has no usable history.
    -> (statements, lines Yahoo filled)."""
    a = (sec or {}).get("annual") or {} if (sec or {}).get("ok") else {}
    out, filled = {}, []
    for k in set(SEC_STMT_LINES) | set(yahoo or {}):
        s, y = a.get(k), (yahoo or {}).get(k)
        if s is not None and len(s) >= 2:
            out[k] = s
        elif y is not None and len(y):
            out[k] = y
            if a and k in SEC_CORE_LINES:
                filled.append(k)
        elif s is not None:
            out[k] = s
    if a:
        lat = sec.get("latest") or {}
        out["latest_bs"] = {k: lat.get(k) for k in ("debt", "cash", "equity")}
        if sec.get("shares_now"):
            out["shares_out"] = pd.Series([float(sec["shares_now"])])
    return out, sorted(filled)


@st.cache_data(ttl=900, show_spinner=False)
def sec_batch(syms: tuple, contact: str) -> dict:
    """SEC data for every stock, cached 15 min (shared by both dot styles)."""
    if not valid_contact(contact):
        return {}
    out = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(sec_company, s, contact): s for s in syms}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


def _md(d: str, fmt: str = "%b %d") -> str:
    try:
        return dt.date.fromisoformat(d).strftime(fmt)
    except (TypeError, ValueError):
        return d or "?"


def sec_line(sec: dict, filled: list | None = None) -> str:
    """One caption line: which filings the numbers come from (linked) + recent 8-Ks."""
    if not sec:
        return ""
    if not sec.get("ok"):
        return f"📄 SEC data unavailable ({sec.get('error', 'unknown error')}) — Yahoo numbers used"
    f = sec.get("filings") or {}
    lat = f.get("latest") or {}
    bits = []
    k, q = lat.get("10-K"), lat.get("10-Q")
    if k:
        bits.append(f"[10-K]({k['url']}) FY to {_md(k['period'], '%b %Y')}")
    if q and (not k or q["filed"] > k["filed"]):
        bits.append(f"[10-Q]({q['url']}) qtr to {_md(q['period'], '%b %d, %Y')}")
    if sec.get("years"):
        bits.append(f"{sec['years']} yrs of annual data"
                    + (f" ({sec['basis_note']})" if sec.get("basis_note") else ""))
    st_ = sec.get("stale")
    if st_:
        end = (sec.get("latest") or {}).get("end")
        bits.append(f"⏳ SEC hasn't processed the {_md(st_['filed'])} {st_['form']} yet"
                    + (f" — latest numbers are to {_md(end, '%b %d')}" if end else ""))
    if filled:
        bits.append("Yahoo filled: " + ", ".join(filled))
    ev = f.get("events") or []
    if ev:
        bits.append("8-K: " + ", ".join(f"[{e['label']}]({e['url']}) {_md(e['date'])}" for e in ev[:4]))
    for w in (f.get("watch") or [])[:2]:
        bits.append(f"⚠ [{w['label']}]({w['url']}) {_md(w['date'])}")
    return "📄 SEC: " + " · ".join(bits)


def fetch_bundle(sym: str, include_stmts: bool = False, sec: dict | None = None) -> dict:
    """Fundamentals + analyst + events for one stock. Never raises. With SEC data in
    hand, Yahoo's statements are only pulled to fill lines the filings don't cover."""
    out = {"info": {}, "earnings": None, "news": [], "stmts": {}, "sec": sec, "stmt_filled": []}
    try:
        t = yf.Ticker(sym)
    except Exception as exc:
        out["error"] = str(exc)
        return out
    # Each piece is fetched independently: Yahoo can block one endpoint
    # (e.g. company info) while statements, calendar and news still work.
    try:
        out["info"] = t.info or {}
    except Exception as exc:
        out["error"] = str(exc)
    if include_stmts:
        yahoo = {}
        if sec_needs_yahoo(sec):
            try:
                yahoo = extract_stmt_data(t)
            except Exception:
                pass
        out["stmts"], out["stmt_filled"] = merge_stmts(sec, yahoo)
    try:
        cal = t.calendar or {}
        ed = cal.get("Earnings Date")
        if isinstance(ed, (list, tuple)) and ed:
            ed = ed[0]
        if ed is not None:
            out["earnings"] = pd.Timestamp(ed).date()
    except Exception:
        pass
    try:
        out["news"] = parse_news_items(t.news)
    except Exception:
        pass
    return out


def financial_health(info: dict) -> tuple[int, list[str]]:
    """Score recent-financials checks: +1 healthy, -1 weak, 0 mixed/unknown."""
    checks, notes = [], []
    rg = info.get("revenueGrowth")
    if rg is not None:
        checks.append(rg > 0)
        notes.append(f"rev {rg * 100:+.0f}% YoY")
    pm = info.get("profitMargins")
    if pm is not None:
        checks.append(pm > 0.05)
        notes.append(f"margin {pm * 100:.0f}%")
    roe = info.get("returnOnEquity")
    if roe is not None:
        checks.append(roe > 0.10)
    de = info.get("debtToEquity")
    if de is not None:
        checks.append(de < 150)
    fcf = info.get("freeCashflow")
    if fcf is not None:
        checks.append(fcf > 0)
        if fcf <= 0:
            notes.append("negative FCF")
    if len(checks) < 3:
        return 0, notes
    passed = sum(checks)
    score = 1 if passed >= max(3, len(checks) - 1) else -1 if passed <= len(checks) // 2 - 1 else 0
    return score, notes


def analyst_view(info: dict) -> tuple[int, str | None]:
    mean = info.get("recommendationMean")
    key = (info.get("recommendationKey") or "").replace("_", " ")
    n = info.get("numberOfAnalystOpinions")
    if mean is None:
        return 0, None
    score = 1 if mean <= 2.5 else -1 if mean >= 3.5 else 0
    label = f"{key or 'rating'} {mean:.1f}" + (f" ({n} analysts)" if n else "")
    target, price = info.get("targetMeanPrice"), info.get("regularMarketPrice") or info.get("currentPrice")
    if target and price:
        label += f", target {((target / price) - 1) * 100:+.0f}%"
    return score, label


def _stmt_row(df, *names):
    """First matching row from a yfinance statement DataFrame, as floats."""
    for n in names:
        try:
            if df is not None and n in df.index:
                s = df.loc[n].dropna()
                if len(s):
                    return s.astype(float)
        except Exception:
            continue
    return None


def _sum_rows(*series):
    """Add statement rows that may be missing or cover different years."""
    got = [s for s in series if s is not None and len(s)]
    if not got:
        return None
    out = got[0]
    for s in got[1:]:
        out = out.add(s, fill_value=0)
    return out.sort_index(ascending=False)


def extract_stmt_data(t) -> dict:
    """Pull the annual-statement lines Compounder mode needs (newest year first)."""
    out = {}
    try:
        inc, bal, cf = t.income_stmt, t.balance_sheet, t.cashflow
        out["revenue"] = _stmt_row(inc, "Total Revenue")
        out["gross"] = _stmt_row(inc, "Gross Profit")
        out["ebit"] = _stmt_row(inc, "EBIT", "Operating Income")
        out["op_income"] = _stmt_row(inc, "Operating Income", "EBIT")
        out["net_income"] = _stmt_row(inc, "Net Income")
        out["shares"] = _stmt_row(inc, "Diluted Average Shares", "Basic Average Shares")
        out["shares_out"] = _stmt_row(bal, "Ordinary Shares Number", "Share Issued")
        out["equity"] = _stmt_row(bal, "Stockholders Equity", "Common Stock Equity",
                                  "Total Equity Gross Minority Interest")
        out["debt"] = _stmt_row(bal, "Total Debt", "Long Term Debt")
        out["cash"] = _stmt_row(bal, "Cash And Cash Equivalents",
                                "Cash Cash Equivalents And Short Term Investments")
        out["deferred_rev"] = _sum_rows(_stmt_row(bal, "Current Deferred Revenue"),
                                        _stmt_row(bal, "Non Current Deferred Revenue"))
        out["fcf"] = _stmt_row(cf, "Free Cash Flow")
        out["ocf"] = _stmt_row(cf, "Operating Cash Flow")
        out["sbc"] = _stmt_row(cf, "Stock Based Compensation")
    except Exception:
        pass
    return out


def _cagr(s: pd.Series) -> float | None:
    """Annualized growth of a newest-first statement series."""
    s = s.dropna() if s is not None else None
    if s is None or len(s) < 2 or s.iloc[-1] <= 0 or s.iloc[0] <= 0:
        return None
    try:   # actual span between fiscal years (a missing year doesn't skew it)
        years = (pd.Timestamp(s.index[0]) - pd.Timestamp(s.index[-1])).days / 365.25
    except (TypeError, ValueError):
        years = 0
    if years < 0.9:
        years = len(s) - 1
    return (s.iloc[0] / s.iloc[-1]) ** (1 / years) - 1


# Compounder thresholds (one place to tune them)
MARGIN_PEAK_PP = 0.10       # current op margin this far above its multi-year average -> peak-cycle risk
MARGIN_STABLE_PP = 0.05     # within this band of the average -> stable
SBC_HEAVY = 0.15            # stock comp above this share of revenue -> FCF overstated
PREPAY_HEAVY = 0.20         # one year's rise in customer prepayments above this share of operating cash flow
EV_FCF_CHEAP, EV_FCF_RICH = 20, 35   # EV / normalized FCF: <=20x cheap-ish (~5% yield), >35x expensive


def _pillar_dot(v: int) -> str:
    return "🟢" if v > 0 else "🔴" if v < 0 else "🟡"


def compounder_signals(stmts: dict, info: dict, price: float | None = None):
    """Long-hold scoring on three questions — is the business healthy, are its
    profits durable, is the price reasonable. Each check scores +1/0/-1.
    -> (parts, score, used, pillars)."""
    pillars = {"health": 0, "durable": 0, "price": 0}
    parts, score, used = [], 0, 0

    def add(pillar, pts, text):
        nonlocal score, used
        pillars[pillar] += pts
        score += pts
        used += 1
        parts.append(text)

    rev, ni = stmts.get("revenue"), stmts.get("net_income")
    op = stmts.get("op_income") if stmts.get("op_income") is not None else stmts.get("ebit")
    ebit = stmts.get("ebit") if stmts.get("ebit") is not None else op
    shares, equity = stmts.get("shares"), stmts.get("equity")
    debt, cash, fcf = stmts.get("debt"), stmts.get("cash"), stmts.get("fcf")
    sbc, ocf, dref = stmts.get("sbc"), stmts.get("ocf"), stmts.get("deferred_rev")
    lb = stmts.get("latest_bs") or {}     # newest balance sheet (10-Q) when SEC data is used

    def _now(key, series):
        v = lb.get(key)
        if v is not None:
            return float(v)
        return float(series.iloc[0]) if series is not None and len(series) else None
    d0, c0, e0 = _now("debt", debt), _now("cash", cash), _now("equity", equity)

    # ── HEALTH ─────────────────────────────────────────────────────────
    # 1. ROIC every year, not just the best one: median AND worst year
    if ebit is not None and equity is not None:
        idx = ebit.index.intersection(equity.index)
        if len(idx):
            d = debt.reindex(idx).fillna(0) if debt is not None else 0
            c = cash.reindex(idx).fillna(0) if cash is not None else 0
            invested = equity.reindex(idx) + d - c
            roics = (ebit.reindex(idx) / invested)[invested > 0].dropna()
            if len(roics):
                med, worst = float(roics.median()), float(roics.min())
                pts = 1 if (med >= 0.15 and worst >= 0.08) else -1 if med < 0.08 else 0
                add("health", pts, f"ROIC median {med * 100:.0f}% (worst yr {worst * 100:.0f}%)")

    # 2. Balance sheet: net debt vs equity (latest quarter when available)
    if d0 is not None and e0 is not None:
        cash_now = c0 or 0.0
        if cash_now >= d0:
            add("health", 1, "net cash")
        elif e0 > 0:
            nd = (d0 - cash_now) / e0
            add("health", 1 if nd <= 0.5 else 0 if nd <= 1.5 else -1, f"net debt {nd:.1f}x equity")
        else:
            add("health", -1, "negative equity")

    # ── DURABILITY ─────────────────────────────────────────────────────
    # 3. Cycle check: current op margin vs its own multi-year average
    if op is not None and rev is not None:
        idx = op.index.intersection(rev.index)
        m = (op.reindex(idx) / rev.reindex(idx)).dropna()
        m = m[rev.reindex(m.index) > 0].sort_index(ascending=False)
        if len(m) >= 3:
            cur, avg = float(m.iloc[0]), float(m.mean())
            gap = cur - avg
            label = f"op margin {cur * 100:.0f}% vs {len(m)}-yr avg {avg * 100:.0f}%"
            if gap > MARGIN_PEAK_PP:
                add("durable", -1, label + " ⚠ far above normal — mean-reversion risk")
            elif abs(gap) <= MARGIN_STABLE_PP:
                add("durable", 1, label + " (steady)")
            else:
                add("durable", 0, label + (" (below normal)" if gap < 0 else ""))

    # 4. FCF quality: FCF minus stock comp ("real FCF") vs earnings, plus prepayment check
    real_fcf = None
    if fcf is not None and len(fcf):
        sbc_al = sbc.reindex(fcf.index).fillna(0) if sbc is not None else 0
        real_fcf = (fcf - sbc_al).dropna().sort_index(ascending=False)
        ni_sum = float(ni.reindex(real_fcf.index).sum()) if ni is not None else 0.0
        ratio = float(real_fcf.sum()) / ni_sum if ni_sum > 0 else None
        if float(real_fcf.sum()) <= 0:
            pts, txt = -1, "real FCF (after stock comp) negative"
        elif ratio is not None:
            pts = 1 if (bool((real_fcf > 0).all()) and ratio >= 0.8) else -1 if ratio < 0.4 else 0
            txt = f"real FCF backs {min(ratio, 2.0) * 100:.0f}% of earnings"
        else:
            pts, txt = (1 if bool((real_fcf > 0).all()) else 0), "real FCF positive"
        flags = []
        if sbc is not None and rev is not None and len(sbc) and len(rev) and float(rev.iloc[0]) > 0:
            sbc_pct = float(sbc.iloc[0]) / float(rev.iloc[0])
            if sbc_pct > SBC_HEAVY:
                flags.append(f"stock comp {sbc_pct * 100:.0f}% of revenue")
        if dref is not None and len(dref) >= 2 and ocf is not None and len(ocf) and float(ocf.iloc[0]) > 0:
            bump = float(dref.iloc[0] - dref.iloc[1]) / float(ocf.iloc[0])
            if bump > PREPAY_HEAVY:
                flags.append(f"customer prepayments = {bump * 100:.0f}% of op cash flow")
        if flags:
            pts = min(pts, 0)           # flagged cash flow can't earn the point
            txt += " ⚠ " + ", ".join(flags)
        add("durable", pts, txt)

    # 5. Dilution: share count trend
    sh_cagr = _cagr(shares)
    if sh_cagr is not None:
        add("durable", 1 if sh_cagr <= -0.005 else 0 if sh_cagr <= 0.015 else -1,
            f"shares {sh_cagr * 100:+.1f}%/yr")

    # 6. Revenue growth
    rev_cagr = _cagr(rev)
    if rev_cagr is not None:
        add("durable", 1 if rev_cagr >= 0.08 else 0 if rev_cagr >= 0.02 else -1,
            f"rev {rev_cagr * 100:+.0f}%/yr")

    # ── PRICE ──────────────────────────────────────────────────────────
    # 7. EV / normalized FCF: average real FCF of the last few years, so a peak
    #    (or trough) year can't make the stock look cheap (or expensive)
    mcap = info.get("marketCap")
    if not mcap and price:
        sh_now = stmts.get("shares_out") if stmts.get("shares_out") is not None else shares
        if sh_now is not None and len(sh_now):
            mcap = float(price) * float(sh_now.iloc[0])
    if mcap and real_fcf is not None and len(real_fcf):
        ev = float(mcap) + (d0 or 0.0) - (c0 or 0.0)
        recent = real_fcf.iloc[:PRICE_NORM_YEARS]
        norm = float(recent.mean())
        if ev <= 0:
            add("price", 0, "EV/FCF n/m (cash exceeds market cap + debt — check manually)")
        elif norm <= 0:
            add("price", -1, f"EV/FCF n/m (avg real FCF negative)")
        else:
            mult = ev / norm
            pts = 1 if mult <= EV_FCF_CHEAP else 0 if mult <= EV_FCF_RICH else -1
            add("price", pts, f"EV/FCF {mult:.0f}x on {len(recent)}-yr avg real FCF "
                              f"(yield {norm / ev * 100:.1f}%)")

    if used:
        parts.insert(0, "Health {} · Durable {} · Price {}".format(
            *(_pillar_dot(pillars[k]) for k in ("health", "durable", "price"))))
    return parts, score, used, pillars


def analyze_stock(sym: str, series: pd.Series, bundle: dict, today: dt.date,
                  style: str = "momentum") -> dict:
    """Combine all signals into a dot + short description. style='momentum'
    rewards strength; style='dip' inverts the price-position signals to
    reward quality trading below its own trend."""
    info = bundle.get("info") or {}
    sec = bundle.get("sec") or {}
    name = (info.get("shortName") or info.get("longName") or sec.get("name") or "")[:38]

    price = day_pct = None
    if series is not None and len(series) >= 2:
        live = info.get("regularMarketPrice")
        price, prev = current_and_prev(series, float(live) if live else None, today)
        day_pct = (price - prev) / prev * 100

    macd = macd_state(series) if series is not None else None
    fib = None
    if series is not None and len(series) and price:
        recent = series[series.index >= series.index[-1] - pd.DateOffset(months=6)]
        fib = fib_state(recent, price)
    ma_dist = gap_pos = gap_neg = None
    if series is not None and len(series) >= 100 and price:
        ma100 = series.rolling(100).mean()
        last_ma = float(ma100.iloc[-1])
        if last_ma > 0:
            ma_dist = (price - last_ma) / last_ma * 100
        gaps = ((series - ma100) / ma100 * 100).dropna()
        gaps = gaps[gaps.index >= series.index[-1] - pd.DateOffset(months=6)]
        pos, neg = gaps[gaps > 0], gaps[gaps < 0]
        gap_pos = float(pos.mean()) if len(pos) else None
        gap_neg = float(neg.mean()) if len(neg) else None
    fin_in = dict(info)      # SEC's latest-quarter / trailing-12-month figures win over Yahoo's
    if sec.get("ok"):
        fin_in.update({k: v for k, v in (sec.get("latest") or {}).items()
                       if k in HEALTH_KEYS and v is not None})
    fin_score, fin_notes = financial_health(fin_in)
    an_score, an_label = analyst_view(info)

    if style == "compounder":
        parts, score, used, pillars = compounder_signals(bundle.get("stmts") or {}, info, price)
    else:
        parts, score, used, pillars = [], 0, 0, {}
        if an_label:
            parts.append(an_label)
            score += an_score
            used += 1
        if macd:
            parts.append("MACD bullish" + (" ↑" if macd["rising"] else "") if macd["bullish"] else "MACD bearish")
            score += 1 if macd["bullish"] else -1
            used += 1
        if ma_dist is not None:
            avg_txt = ""
            if gap_pos is not None or gap_neg is not None:
                p_txt = f"+{gap_pos:.1f}%" if gap_pos is not None else "—"
                n_txt = f"{gap_neg:.1f}%" if gap_neg is not None else "—"
                avg_txt = f" (6-mo avg {p_txt} / {n_txt})"
            parts.append(f"{ma_dist:+.1f}% vs MA100{avg_txt}")
            ma_pt = 1 if ma_dist > 1 else -1 if ma_dist < -1 else 0
            score += -ma_pt if style == "dip" else ma_pt
            used += 1
        if fib:
            parts.append(f"{fib['pct'] * 100:.0f}% of 6-mo range (fib {fib['zone']})")
            fib_pt = 1 if fib["zone"] == "strong" else -1 if fib["zone"] == "weak" else 0
            score += -fib_pt if style == "dip" else fib_pt
            used += 1
        if fin_notes or fin_score:
            parts.append(", ".join(fin_notes) if fin_notes else "financials mixed")
            score += fin_score
            used += 1

    earnings = bundle.get("earnings")
    warn = ""
    if earnings and 0 <= (earnings - today).days <= 7:
        warn = f" ⚠️ earnings {earnings:%b %d}"
    elif earnings and (earnings - today).days > 0:
        parts.append(f"earnings {earnings:%b %d}")

    # SEC red-flag filings (bankruptcy, debt default, restatement, late filing) in the
    # past year force a red dot whatever the numbers say
    flags = ((sec.get("filings") or {}).get("flags") or []) if sec.get("ok") else []
    if flags:
        parts.insert(0, "🚩 " + ", ".join(f"{f['label']} ({_md(f['date'], '%b %d, %Y')})" for f in flags[:2]))

    red_pillar = any(v < 0 for v in pillars.values())  # Compounder: a red pillar blocks green
    dot = RED_DOT if flags else GRAY_DOT if used < 2 else \
        GREEN_DOT if (score >= MIN_SCORE_FOR_GREEN and not red_pillar) else RED_DOT
    return {
        "sym": sym, "name": name, "price": price, "day_pct": day_pct,
        "dot": dot, "score": score, "desc": " · ".join(parts) or "insufficient data",
        "warn": warn, "news": bundle.get("news") or [],
        "sec_line": sec_line(sec, bundle.get("stmt_filled")),
    }


SOURCE_NOTES = {
    "live": "Top 5 from Yahoo's live fund data",
    "cache": "Top 5 from the last live list, saved {date}",
    "mcap": "Top 5 rebuilt by live market cap — the fund weights by size, so these are its biggest holdings",
    "static": "Top 5 from the built-in list (live market caps unavailable right now)",
}


def fmt_cap(v: float) -> str:
    return f"${v / 1e12:.2f}T" if v >= 1e12 else f"${v / 1e9:.0f}B"


def holdings_banner(sources: dict) -> str | None:
    """One plain-language note when any sector isn't on live fund data."""
    kinds = [k for k, _ in sources.values()]
    if not kinds or all(k == "live" for k in kinds):
        return None
    n = lambda k: kinds.count(k)
    bits = []
    if n("mcap"):
        bits.append(f"{n('mcap')} rebuilt from live market caps")
    if n("cache"):
        bits.append(f"{n('cache')} from the last saved live list")
    if n("static"):
        bits.append(f"{n('static')} from the built-in list")
    return ("ℹ️ Yahoo is currently blocking its ETF-holdings data (a change on Yahoo's side — "
            "updating yfinance doesn't fix it), so the app built the top-5 lists itself: "
            + ", ".join(bits) + ". Scores and dots work as normal; only fund weights are missing. "
            "It switches back to live data automatically once Yahoo allows it.")


@st.cache_data(ttl=900, show_spinner=False)
def deep_fetch(etfs: tuple, include_stmts: bool = False, sec_contact: str = ""):
    """Holdings + price history + fundamentals for every top-5 stock. Cached 15 min.
    sources[etf] = ("live" | "cache" | "mcap" | "static", date or None)."""
    today = dt.date.today()
    holdings, sources, errors, need_rank = resolve_holdings(etfs, today)

    closes, caps = pd.DataFrame(), {}
    if need_rank:
        # One 1-year price pull for every candidate: it ranks them by market
        # cap now AND supplies the analysis history afterwards.
        cands = sorted({s for etf in need_rank for s in SECTOR_CANDIDATES[etf]})
        closes = fetch_history(cands, period="1y")
        caps = market_caps_from(candidate_shares(tuple(cands)), closes)
        for etf in need_rank:
            top = rank_by_market_cap(SECTOR_CANDIDATES[etf], caps)
            if top:
                holdings[etf], sources[etf] = [(s, None) for s in top], ("mcap", None)
            else:
                holdings[etf] = [(s, None) for s in FALLBACK_HOLDINGS[etf]]
                sources[etf] = ("static", None)

    syms = sorted({s for lst in holdings.values() for s, _ in lst})
    missing = [s for s in syms if s not in closes.columns]
    if missing:
        more = fetch_history(missing, period="1y")
        closes = more if closes.empty else closes.join(more, how="outer")
    sec = sec_batch(tuple(syms), sec_contact) if syms and valid_contact(sec_contact) else {}
    bundles = {}
    if syms:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(fetch_bundle, s, include_stmts, sec.get(s)): s for s in syms}
            for fut in as_completed(futures):
                bundles[futures[fut]] = fut.result()
    return holdings, errors, sources, closes, bundles, caps


# ═══════════════════════ interactive chart ═══════════════════════

def flatten_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to plain Open/High/Low/Close columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "High", "Low", "Close"])


@st.cache_data(ttl=300, show_spinner=False)
def chart_data(sym: str, period: str) -> pd.DataFrame:
    raw = yf.download(sym, period=period, interval="1d",
                      auto_adjust=False, progress=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    return flatten_ohlc(raw)


def make_chart(ohlc: pd.DataFrame, sym: str, months_to_show: int | None = None) -> go.Figure:
    """Candles + 100-day MA. The MA is computed on the full fetched history,
    then both are trimmed to the display window so the MA appears even on
    short ranges."""
    ma = ohlc["Close"].rolling(100).mean() if len(ohlc) >= 100 else None
    view = ohlc
    if months_to_show:
        cutoff = ohlc.index[-1] - pd.DateOffset(months=months_to_show)
        view = ohlc[ohlc.index >= cutoff]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["Open"], high=view["High"],
        low=view["Low"], close=view["Close"], name=sym,
        increasing_line_color="#1a9c56", decreasing_line_color="#d64545",
    ))
    if ma is not None:
        ma_view = ma.loc[view.index]
        if ma_view.notna().any():
            fig.add_trace(go.Scatter(x=ma_view.index, y=ma_view, name="100-day MA",
                                     line=dict(color="#f59e0b", width=1.5)))
    fig.update_layout(
        height=520, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.02),
        newshape=dict(line=dict(color="#2563eb", width=2)),  # style of drawn lines
    )
    return fig


DRAW_CONFIG = {
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
    "displayModeBar": True,
    "displaylogo": False,
}


# ═══════════════════════ portfolio tracker ═══════════════════════

PORTFOLIO_FILE = Path(__file__).with_name("portfolio.csv")
PORT_COLUMNS = ["Ticker", "Sector", "Shares", "Price", "Day %", "Day P/L",
                "Value", "Weight %", "Total P/L", "Total P/L %"]

TICKER_RE = re.compile(r"[A-Z][A-Z0-9.\-/]{0,9}")
SKIP_ROWS = ("POSITIONS TOTAL", "ACCOUNT TOTAL", "TOTAL")


def _money(v) -> float:
    """'$1,307.79 ', '($43.04)', '15.08%', '--', 'N/A' -> float or NaN."""
    s = str(v).strip().replace("$", "").replace(",", "").replace("%", "").strip()
    if s in ("", "--", "N/A", "nan", "None"):
        return float("nan")
    neg = s.startswith("(") and s.endswith(")")
    try:
        x = float(s.strip("()"))
    except ValueError:
        return float("nan")
    return -x if neg else x


def parse_schwab_csv(text: str):
    """Schwab 'Positions' website export. No share column, so shares are derived:
    market value = cost basis + gain $, shares = market value / price."""
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines)
               if l.replace('"', "").split(",")[0].strip() == "Symbol")
    m = re.search(r"as of (.+?)[\",]*$", lines[0]) if hdr > 0 else None
    asof = m.group(1).strip() if m else None

    df = pd.read_csv(io.StringIO(text), skiprows=hdr)
    df.columns = [str(c).strip() for c in df.columns]
    col = lambda prefix: next(c for c in df.columns if c.startswith(prefix))
    cost_c, price_c, gain_c, pct_c = col("Cost Basis"), col("Price"), col("Gain $"), col("% of Acct")

    rows, cash_pct = [], None
    for _, r in df.iterrows():
        sym = str(r["Symbol"]).strip()
        if sym.upper().startswith("CASH"):
            cash_pct = _money(r[pct_c])
            continue
        if sym.upper() in SKIP_ROWS or not TICKER_RE.fullmatch(sym):
            continue
        cost, price, gain = _money(r[cost_c]), _money(r[price_c]), _money(r[gain_c])
        if pd.isna(price) or price <= 0 or (pd.isna(cost) and pd.isna(gain)):
            continue
        mv = (0 if pd.isna(cost) else cost) + (0 if pd.isna(gain) else gain)
        shares = mv / price
        if shares <= 0:
            continue
        rows.append({"ticker": sym.upper(), "shares": shares,
                     "cost": cost / shares if pd.notna(cost) else float("nan"),
                     "_mv": mv})

    pf = pd.DataFrame(rows)
    sec_mv = float(pf["_mv"].sum()) if len(pf) else 0.0
    if len(pf):
        pf = pf.drop(columns="_mv")
    cash_value = cash_from_pct(sec_mv, cash_pct)
    return pf, cash_value, asof


def cash_from_pct(securities_mv: float, cash_pct: float | None) -> float | None:
    """Schwab gives cash only as % of account; back out its dollar value."""
    if cash_pct is None or pd.isna(cash_pct) or not (0 < cash_pct < 100):
        return None
    return securities_mv * cash_pct / (100 - cash_pct)


def parse_simple_csv(text: str) -> pd.DataFrame:
    """Hand-kept format: ticker,shares,cost_basis — duplicate lots get merged."""
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "ticker" not in raw.columns or "shares" not in raw.columns:
        raise ValueError("simple portfolio csv needs 'ticker' and 'shares' columns")
    cost_col = next((c for c in ("cost_basis", "cost", "avg_cost") if c in raw.columns), None)
    raw["ticker"] = raw["ticker"].astype(str).str.strip().str.upper()
    raw["shares"] = pd.to_numeric(raw["shares"], errors="coerce").fillna(0)
    raw["cost"] = pd.to_numeric(raw[cost_col], errors="coerce") if cost_col else float("nan")
    raw = raw[raw["ticker"] != ""]
    rows = []
    for tkr, g in raw.groupby("ticker"):
        shares = float(g["shares"].sum())
        lots = g.dropna(subset=["cost"])
        lots = lots[lots["shares"] > 0]
        cost = float((lots["shares"] * lots["cost"]).sum() / lots["shares"].sum()) if len(lots) else float("nan")
        rows.append({"ticker": tkr, "shares": shares, "cost": cost})
    return pd.DataFrame(rows)


def find_portfolio_file() -> Path | None:
    """Newest Schwab '...Positions...csv' export in the script folder wins;
    otherwise fall back to portfolio.csv."""
    folder = Path(__file__).parent
    exports = [p for p in folder.glob("*.csv") if "position" in p.name.lower()]
    if exports:
        return max(exports, key=lambda p: p.stat().st_mtime)
    return PORTFOLIO_FILE if PORTFOLIO_FILE.exists() else None


def parse_portfolio_text(text: str):
    """Route raw CSV text to the right parser. -> (df, cash $ or None, as-of or None)."""
    if re.search(r"^\"?Symbol\"?,", text, flags=re.MULTILINE) and "Cost Basis" in text:
        return parse_schwab_csv(text)
    return parse_simple_csv(text), None, None


def load_any_portfolio():
    """-> (positions df, cash $ or None, source name, as-of string or None)."""
    path = find_portfolio_file()
    if path is None:
        return None, None, None, None
    pf, cash_value, asof = parse_portfolio_text(path.read_text(errors="replace"))
    return pf, cash_value, path.name, asof


@st.cache_data(ttl=300, show_spinner=False)
def portfolio_prices(syms: tuple):
    """1-year closes + live quotes for the portfolio tickers."""
    closes = fetch_history(list(syms), period="1y")
    live = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(get_live_price, s): s for s in syms}
        for fut in as_completed(futures):
            live[futures[fut]] = fut.result()
    return closes, live


SECTOR_KEY_MAP = {
    "technology": "Information Technology", "financial_services": "Financials",
    "consumer_cyclical": "Consumer Discretionary", "consumer_defensive": "Consumer Staples",
    "communication_services": "Communication Services", "healthcare": "Health Care",
    "industrials": "Industrials", "energy": "Energy", "basic_materials": "Materials",
    "real_estate": "Real Estate", "realestate": "Real Estate", "utilities": "Utilities",
}


def normalize_sector_label(label) -> str:
    """Map Yahoo's sector spellings ('Technology', 'financial_services',
    'Consumer Cyclical'…) onto one consistent set of GICS-style names."""
    if not label:
        return "—"
    key = str(label).strip().lower().replace(" ", "_")
    return SECTOR_KEY_MAP.get(key, str(label))


def _ticker_meta(sym: str) -> dict:
    """Sector for stocks; for ETFs also the fund's sector weightings (look-through)."""
    out = {"sector": "—", "weights": None, "category": None}
    try:
        t = yf.Ticker(sym)
        info = t.info or {}
        if info.get("quoteType") == "ETF":
            out["sector"] = "ETF"
            out["category"] = info.get("category")
            try:
                sw = t.funds_data.sector_weightings
                if sw:
                    out["weights"] = dict(sw)
            except Exception:
                pass
        else:
            out["sector"] = normalize_sector_label(info.get("sector") or info.get("quoteType"))
    except Exception:
        pass
    return out


@st.cache_data(ttl=604800, show_spinner=False)
def ticker_meta(syms: tuple) -> dict:
    """Per-ticker metadata, cached a week (sectors and fund mixes barely move)."""
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_ticker_meta, s): s for s in syms}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


def sector_allocation(pdf: pd.DataFrame, meta: dict, cash: float | None) -> pd.Series:
    """Portfolio value per sector. ETF value is split across its underlying
    sector weightings; funds without weightings fall back to their category."""
    alloc: dict[str, float] = {}

    def add(label: str, v: float):
        alloc[label] = alloc.get(label, 0.0) + v

    for _, r in pdf.iterrows():
        m = meta.get(r["Ticker"], {})
        weights = m.get("weights")
        if weights:
            total_w = sum(weights.values()) or 1.0
            for key, frac in weights.items():
                add(normalize_sector_label(key), float(r["Value"]) * float(frac) / total_w)
        elif m.get("sector") not in (None, "—", "ETF"):
            add(normalize_sector_label(m["sector"]), float(r["Value"]))
        else:
            add(m.get("category") or "Other funds", float(r["Value"]))
    if cash:
        add("Cash", float(cash))
    return pd.Series(alloc).sort_values(ascending=False)


def make_alloc_pie(alloc: pd.Series) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=list(alloc.index), values=[float(v) for v in alloc.values],
        hole=0.35, textinfo="percent", textposition="inside", sort=True,
    ))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="v"))
    return fig


def build_portfolio_table(pf: pd.DataFrame, closes: pd.DataFrame, live: dict,
                          sectors: dict, today: dt.date, cash: float | None = None):
    """Per-position live valuation + portfolio totals (cash included in totals)."""
    rows, missing = [], []
    for _, h in pf.iterrows():
        sym, shares, cost = h["ticker"], float(h["shares"]), h["cost"]
        s = closes[sym].dropna() if sym in closes.columns else pd.Series(dtype=float)
        if len(s) < 2:
            missing.append(sym)
            continue
        current, prev = current_and_prev(s, live.get(sym), today)
        day_pct = (current - prev) / prev * 100
        value = shares * current
        tpl = shares * (current - cost) if pd.notna(cost) and cost > 0 else float("nan")
        tpl_pct = (current / cost - 1) * 100 if pd.notna(cost) and cost > 0 else float("nan")
        rows.append({
            "Ticker": sym, "Sector": sectors.get(sym, "—"), "Shares": shares,
            "Price": current, "Day %": day_pct, "Day P/L": shares * (current - prev),
            "Value": value, "Weight %": 0.0, "Total P/L": tpl, "Total P/L %": tpl_pct,
        })
    df = pd.DataFrame(rows, columns=PORT_COLUMNS)
    totals = {"value": 0.0, "day_pl": 0.0, "day_pct": 0.0, "total_pl": 0.0,
              "has_cost": False, "cash": cash if cash and cash > 0 else None}
    if len(df):
        sec_value = float(df["Value"].sum())
        totals["value"] = sec_value + (totals["cash"] or 0.0)
        totals["day_pl"] = float(df["Day P/L"].sum())
        prev_value = totals["value"] - totals["day_pl"]
        totals["day_pct"] = totals["day_pl"] / prev_value * 100 if prev_value else 0.0
        totals["has_cost"] = bool(df["Total P/L"].notna().any())
        totals["total_pl"] = float(df["Total P/L"].sum(skipna=True))
        df["Weight %"] = df["Value"] / totals["value"] * 100 if totals["value"] else 0.0
        df = df.sort_values("Value", ascending=False)
    return df, totals, missing


def style_port_row(row: pd.Series) -> list[str]:
    def c(v):
        return "" if pd.isna(v) else GREEN if v > 0 else RED if v < 0 else ""
    day, tpl = c(row["Day %"]), c(row["Total P/L"])
    return [day, "", "", "", day, day, "", "", tpl, tpl]


def make_port_styler(df: pd.DataFrame):
    return df.style.apply(style_port_row, axis=1).format({
        "Shares": lambda v: "—" if pd.isna(v) else (f"{v:,.0f}" if float(v).is_integer() else f"{v:,.3f}"),
        "Price": "${:,.2f}", "Day %": "{:+.2f}%", "Day P/L": "${:+,.2f}",
        "Value": "${:,.2f}", "Weight %": "{:.1f}%",
        "Total P/L": "${:+,.2f}", "Total P/L %": "{:+.2f}%",
    }, na_rep="—")


@st.cache_data(ttl=300, show_spinner=False)
def sector_fetch():
    """5-year sector history + live quotes, cached 5 min so page interactions stay snappy."""
    history = fetch_history(list(SECTOR_ETFS), period="5y")
    live = {t: get_live_price(t) for t in SECTOR_ETFS}
    return history, live


# ═══════════════════════ app ═══════════════════════

def main():
    st.set_page_config(page_title="Fidelity Sector Scanner", page_icon="📊", layout="wide")

    head_l, head_r = st.columns([2.6, 1])
    with head_l:
        st.title("📊 Fidelity Sector Scanner")
        st.caption("Sectors vs. yesterday, 3-mo, 1-yr & 5-yr · deep scan of top-5 holdings — screening signals, not buy/sell advice")
    with head_r:
        fed = fetch_fed_rate()
        if fed:
            st.metric(f"Fed funds rate ({fed['kind']})", fed["label"])
            st.caption("Higher → savings/bonds pay more, pulls money from stocks; lower → pushes money into stocks")
        else:
            st.caption("Fed rate: unavailable right now")

    top_left, top_right = st.columns([1, 4])
    with top_left:
        if st.button("🔄 Run Scan", type="primary", width="stretch"):
            st.cache_data.clear()  # force fresh data on click
    with top_right:
        sort_by = st.selectbox(
            "Sort", ["Day % (best→worst)", "Day % (worst→best)", "3-Mo % (best→worst)",
                     "1-Yr % (best→worst)", "5-Yr % (best→worst)", "Ticker"],
            index=0, label_visibility="collapsed",
        )

    today = dt.date.today()

    # ---- Part A: sector table ----
    try:
        with st.spinner("Fetching sector prices…"):
            history, live = sector_fetch()
            df = compute_table(history, live, today)
    except Exception as exc:
        st.error(f"Data fetch failed: {exc}\n\nCheck your internet connection and try again.")
        st.stop()

    if df.empty:
        st.warning("No price data returned. Try again in a minute.")
        st.stop()

    # ---- money-flow summary line ----
    greens, reds = top_movers(df)
    if greens and reds:
        st.info(f"💸 Money is moving out of: **{', '.join(reds)}** → into: **{', '.join(greens)}** "
                f"*(based on today's sector moves)*")
    elif greens:
        st.success("🎉 Every sector is green — the market is happy today")
    elif reds:
        st.error("😞 Every sector is red — the market is sad today")
    else:
        st.info("Market is flat today — no sector moved.")

    key = {"Day % (best→worst)": ("Day %", False), "Day % (worst→best)": ("Day %", True),
           "3-Mo % (best→worst)": ("3-Mo %", False), "1-Yr % (best→worst)": ("1-Yr %", False),
           "5-Yr % (best→worst)": ("5-Yr %", False), "Ticker": ("Ticker", True)}[sort_by]
    df = df.sort_values(key[0], ascending=key[1])

    up, down = int((df["Day %"] > 0).sum()), int((df["Day %"] < 0).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Sectors up today", up)
    m2.metric("Sectors down today", down)
    m3.metric("Avg day move", f"{df['Day %'].mean():+.2f}%")
    disp = df.copy()
    disp["Ticker"] = disp["Ticker"].map(lambda t: f"https://finance.yahoo.com/chart/{t}")
    st.dataframe(
        make_styler(disp), width="stretch", hide_index=True, height=430,
        column_config={"Ticker": st.column_config.LinkColumn(
            "Ticker", display_text=r"chart/(.+)$",
            help="Open this ticker's full chart on Yahoo Finance")},
    )

    # ---- my portfolio (live-priced from Schwab export or portfolio.csv) ----
    st.divider()
    show_pf = st.checkbox("💼 My Portfolio — live view of your account", value=False)
    if show_pf:
        uploaded = st.file_uploader(
            "Upload your Schwab Positions export (running in the cloud or on a tablet? "
            "this is your way in — on your PC you can keep dropping the file in the folder instead)",
            type=["csv"],
        )
        try:
            if uploaded is not None:
                pf, cash, asof = parse_portfolio_text(
                    uploaded.getvalue().decode("utf-8", errors="replace"))
                source = uploaded.name
            else:
                pf, cash, source, asof = load_any_portfolio()
        except Exception as exc:
            pf, cash, source, asof = None, None, None, None
            st.error(f"Couldn't read your portfolio file: {exc}")
        if pf is None:
            st.info("Use the upload box above with your Schwab **Positions export** "
                    "(schwab.com → Accounts → Positions → Export) — or, when running on "
                    "your PC, just drop that `...Positions...csv` into the script's folder. "
                    "A simple **portfolio.csv** (`ticker,shares,cost_basis`) works too.")
        elif len(pf) == 0 or (pf["shares"] <= 0).all():
            st.info(f"Found {source} but no usable positions in it — check the file.")
        else:
            pf = pf[pf["shares"] > 0]
            syms = tuple(sorted(pf["ticker"].unique()))
            with st.spinner("Pricing portfolio…"):
                closes_p, live_p = portfolio_prices(syms)
                meta_map = ticker_meta(syms)
            sector_map = {s: meta_map.get(s, {}).get("sector", "—") for s in syms}
            pdf, totals, missing = build_portfolio_table(pf, closes_p, live_p, sector_map, today, cash)
            if pdf.empty:
                st.warning("No price data returned for your holdings — check the ticker symbols.")
            else:
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Total value", f"${totals['value']:,.2f}", f"{totals['day_pct']:+.2f}% today")
                p2.metric("Day P/L", f"${totals['day_pl']:+,.2f}")
                if totals["has_cost"]:
                    p3.metric("Total P/L (vs cost)", f"${totals['total_pl']:+,.2f}")
                if totals["cash"]:
                    p4.metric("Cash", f"${totals['cash']:,.2f}",
                              f"{totals['cash'] / totals['value'] * 100:.1f}% of account",
                              delta_color="off")
                pdisp = pdf.copy()
                pdisp["Ticker"] = pdisp["Ticker"].map(lambda t: f"https://finance.yahoo.com/chart/{t}")
                st.dataframe(
                    make_port_styler(pdisp), width="stretch", hide_index=True,
                    height=min(80 + 36 * len(pdisp), 560),
                    column_config={"Ticker": st.column_config.LinkColumn(
                        "Ticker", display_text=r"chart/(.+)$", help="Open on Yahoo Finance")},
                )
                if missing:
                    st.caption("⚠ No data for: " + ", ".join(missing))

                st.markdown("**🥧 Sector diversification** — ETFs are looked through to "
                            "their underlying sector mix, so funds like QQQM count toward "
                            "the sectors they actually hold")
                alloc = sector_allocation(pdf, meta_map, totals["cash"])
                if len(alloc):
                    st.plotly_chart(make_alloc_pie(alloc), width="stretch",
                                    config={"displayModeBar": False})
                    top_pct = float(alloc.iloc[0]) / float(alloc.sum()) * 100
                    st.caption(f"Spread across {len(alloc)} slices · biggest concentration: "
                               f"{alloc.index[0]} at {top_pct:.1f}% of the account")
            src_txt = f"Source: {source}" + (f" (Schwab export as of {asof})" if asof else "")
            st.caption(f"{src_txt} — download a fresh Positions export from Schwab and drop it in "
                       "the folder to update. Prices are live; nothing is sent anywhere.")

    # ---- Part B: deep scan ----
    st.divider()
    run_deep = st.checkbox("🔎 Deep scan — analyze top-5 holdings of each sector", value=False)
    if run_deep:
        contact = st.session_state.get("sec_contact") or sec_contact()
        if not valid_contact(contact):
            with st.container(border=True):
                st.markdown("**📄 Connect SEC EDGAR** — score stocks on the numbers companies "
                            "file with the SEC (10-K / 10-Q), the original source behind Yahoo's data. "
                            "Free, no account needed.")
                c1, c2 = st.columns([3, 1])
                entered = c1.text_input("Your name and email", placeholder="Your Name you@example.com",
                                        label_visibility="collapsed")
                if c2.button("Connect", width="stretch"):
                    if valid_contact(entered):
                        st.session_state["sec_contact"] = entered.strip()
                        save_contact(entered)
                        st.rerun()
                    else:
                        st.warning("Enter a name and a valid email, e.g. `Jane Doe jane@example.com`.")
                st.caption("SEC asks every app that reads EDGAR to identify itself with a name + email. "
                           "It's sent only to sec.gov, in the request header, and saved in "
                           "sec_contact.txt next to this script. Until it's set, the deep scan uses "
                           "Yahoo's numbers.")
        style_pick = st.radio(
            "🟢 Dot style",
            ["Compounder — long holds: healthy, durable, fairly priced",
             "Dip-buyer — shorter holds: quality trading below its trend"],
            horizontal=True,
        )
        style = "dip" if style_pick.startswith("Dip") else "compounder"
        chosen = st.multiselect(
            "Sectors to analyze", list(SECTOR_ETFS),
            default=list(SECTOR_ETFS),
            format_func=lambda t: f"{t} · {SECTOR_ETFS[t]}",
        )
        if chosen:
            with st.spinner("Deep scan running — first pass takes 1-2 minutes (the first SEC run "
                            "also downloads each company's filing history, saved for next time), "
                            "then it's cached…"):
                holdings, errors, sources, closes, bundles, caps = deep_fetch(
                    tuple(chosen), include_stmts=(style == "compounder"),
                    sec_contact=contact if valid_contact(contact) else "")

            banner = holdings_banner(sources)
            if banner:
                st.info(banner)
            if valid_contact(contact) and bundles:
                n_ok = sum(1 for b in bundles.values() if (b.get("sec") or {}).get("ok"))
                st.caption(f"📄 Financials from SEC EDGAR filings for {n_ok}/{len(bundles)} stocks"
                           + ("" if n_ok == len(bundles) else " — the rest use Yahoo's numbers") +
                           ". Each stock's 📄 line links the exact 10-K/10-Q used. Prices, analyst "
                           "ratings and news still come from Yahoo.")
            for etf in chosen:
                sector = SECTOR_ETFS[etf]
                if etf in errors:
                    with st.expander(f"{etf} · {sector}"):
                        st.warning(f"Top holdings unavailable: {errors[etf]}")
                    continue
                results = []
                for sym, w in holdings.get(etf, []):
                    series = closes[sym].dropna() if sym in closes.columns else None
                    r = analyze_stock(sym, series, bundles.get(sym, {}), today, style)
                    r["weight"] = w
                    r["mcap"] = caps.get(sym)
                    results.append(r)
                greens = sum(1 for r in results if r["dot"] == GREEN_DOT)
                with st.expander(f"{etf} · {sector} — {greens}/{len(results)} green"):
                    st.caption(SOURCE_NOTES[sources.get(etf, ("live", None))[0]].format(
                        date=sources.get(etf, (None, ""))[1] or ""))
                    for r in results:
                        if r["weight"]:
                            w_txt = f" ({r['weight'] * 100:.0f}% of fund)"
                        elif r["mcap"]:
                            w_txt = f" (mkt cap {fmt_cap(r['mcap'])})"
                        else:
                            w_txt = ""
                        px = f"${r['price']:,.2f} ({r['day_pct']:+.2f}%)" if r["price"] else "price n/a"
                        st.markdown(f"{r['dot']} **{r['sym']}** {r['name']}{w_txt} — {px}{r['warn']}")
                        st.caption(r["desc"])
                        if r["sec_line"]:
                            st.caption(r["sec_line"])
                        if r["news"]:
                            st.caption("📰 " + "  |  ".join(r["news"]))

    # ---- interactive chart with drawing tools ----
    st.divider()
    show_chart = st.checkbox("📈 Chart — pick a ticker and draw on it", value=False)
    if show_chart:
        c1, c2, c3 = st.columns([2, 2, 1])
        pick = c1.selectbox("Sector ETF", list(SECTOR_ETFS),
                            format_func=lambda t: f"{t} · {SECTOR_ETFS[t]}")
        custom = c2.text_input("…or type any ticker", placeholder="AAPL, NVDA, MSFT…")
        rng = c3.selectbox("Range", ["3mo", "6mo", "1y", "5y"], index=1)
        sym = custom.strip().upper() or pick
        fetch_period, show_months = {"3mo": ("1y", 3), "6mo": ("1y", 6),
                                     "1y": ("2y", 12), "5y": ("5y", None)}[rng]
        ohlc = chart_data(sym, fetch_period)
        if ohlc.empty:
            st.warning(f"No chart data found for '{sym}' — double-check the ticker symbol.")
        else:
            st.plotly_chart(make_chart(ohlc, sym, show_months),
                            width="stretch", config=DRAW_CONFIG)
            st.caption("🖊️ Drawing toolbar is at the top-right of the chart: straight line, freehand, "
                       "rectangle, eraser. Drawings are temporary — they clear when the page reruns. "
                       "For charts you can save, click any ticker in the table (opens Yahoo Finance).")

    st.caption(f"Last updated: {dt.datetime.now():%Y-%m-%d %H:%M:%S} · Data: prices, analyst ratings "
               "& news — Yahoo Finance · financials & filings — SEC EDGAR · Fed rate — FRED")


if __name__ == "__main__":
    main()
