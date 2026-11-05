"""
Kindle market board: plain HTML pages (no JavaScript) for the Kindle browser,
built from sector_scanner.py itself, so every number, dot and description
matches the desktop scanner exactly.

  index.html        sector table, Fed rate, money-flow line, best Compounders / Dip-buys
  s/<ETF>.html      deep scan of the sector's top-5 holdings (both dot styles) + chart
  t/<TICKER>.html   chart (price, MA100, Fibonacci, MACD) + the stock's full scan

Keep sector_scanner.py in the same folder. Then:
    pip install yfinance pandas matplotlib streamlit plotly requests
    python kindle_page.py          # live data (set SEC_CONTACT for SEC financials)
    python kindle_page.py --demo   # made-up data, to preview the layout offline
"""
import argparse
import datetime as dt
import html
import io
import os
import re

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")   # quiet "no runtime" notices

import numpy as np
import pandas as pd

import sector_scanner as sc

OUT = "site"
REFRESH_SECONDS = 900
FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
SHORT = {"Information Technology": "Technology", "Communication Services": "Communication",
         "Consumer Discretionary": "Cons. Discretionary", "Consumer Staples": "Cons. Staples"}


# ---------------------------------------------------------------- text helpers
# E-ink and the Kindle font can't show color emoji: turn them into words.
EMOJI_WORDS = [("ℹ️", "Note:"), ("ℹ", "Note:"), ("⚠️", "(!)"), ("⚠", "(!)"), ("🚩", "RED FLAG:"), ("🏛️", "[policy]"),
               ("🏛", "[policy]"), ("📅", "[earnings]"), ("⏳", "(pending)"), ("📄", ""),
               ("📰", ""), ("🟢", "good"), ("🟡", "ok"), ("🔴", "weak"), ("⚪", "")]
EMOJI_RE = re.compile("[\U0001F000-\U0001FFFF☀-➿️]")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def clean(s):
    for a, b in EMOJI_WORDS:
        s = s.replace(a, b)
    return EMOJI_RE.sub("", s).strip()


def text_html(s):
    """Scanner text -> HTML: emoji to words, markdown links to real links."""
    s, out, pos = clean(s or ""), [], 0
    for m in LINK_RE.finditer(s):
        out.append(html.escape(s[pos:m.start()]))
        out.append(f'<a href="{html.escape(m.group(2))}">{html.escape(m.group(1))}</a>')
        pos = m.end()
    out.append(html.escape(s[pos:]))
    return "".join(out)


def bad(v):
    return v is None or (isinstance(v, (int, float)) and not np.isfinite(v))


def pct(v):
    return "&mdash;" if bad(v) else f"{v:+.1f}%"


def day_html(v):
    if bad(v):
        return "&mdash;"
    v = round(v, 1) + 0.0          # -0.04% counts as flat, not a down day
    arrow = "&#9650;" if v > 0 else "&#9660;" if v < 0 else ""
    text = f"{arrow} {v:+.1f}%"
    return f'<span class="neg">{text}</span>' if v < 0 else text


def dot_tag(r):
    cls, word = {sc.GREEN_DOT: ("strong", "GREEN"), sc.RED_DOT: ("weak", "RED")}.get(r["dot"], ("na", "NO DATA"))
    score = f" {r['score']:+d}" if cls != "na" else ""
    return f'<span class="tag {cls}">{word}{score}</span>'


# ---------------------------------------------------------------- data (via the scanner)
def gather():
    today = dt.date.today()
    history, live = sc.sector_fetch()
    table = sc.compute_table(history, live, today)
    num = [c for c in table.columns if c not in ("Ticker", "Sector")]
    table[num] = table[num].apply(pd.to_numeric, errors="coerce")
    if table.empty:
        return None
    contact = sc.sec_contact()
    holdings, errors, sources, closes, bundles, caps = sc.deep_fetch(
        tuple(sc.SECTOR_ETFS), include_stmts=True,
        sec_contact=contact if sc.valid_contact(contact) else "")

    stocks = {}
    for etf in sc.SECTOR_ETFS:
        for sym, weight in holdings.get(etf, []):
            if sym in stocks:
                continue
            series = closes[sym].dropna() if sym in closes.columns else None
            bundle = bundles.get(sym, {})
            comp = sc.analyze_stock(sym, series, bundle, today, "compounder")
            dip = sc.analyze_stock(sym, series, bundle, today, "dip")
            stocks[sym] = {"sym": sym, "etf": etf, "name": comp["name"], "price": comp["price"],
                           "day_pct": comp["day_pct"], "weight": weight, "mcap": caps.get(sym),
                           "comp": comp, "dip": dip, "warn": comp["warn"], "news": comp["news"],
                           "sec_line": comp["sec_line"], "series": series}
    return {"today": today, "history": history, "table": table, "fed": sc.fetch_fed_rate(),
            "holdings": holdings, "errors": errors, "sources": sources, "stocks": stocks,
            "sec_on": sc.valid_contact(contact)}


def install_demo():
    """Swap the scanner's three network calls for made-up data; everything else
    (scoring, dots, descriptions) runs through the real scanner code."""
    rng = np.random.default_rng(7)
    today = dt.date.today()

    def walk(n, start):
        return start * np.exp(np.cumsum(rng.normal(0.0004, rng.uniform(0.01, 0.022), n)))

    idx5 = pd.bdate_range(end=today, periods=1265)
    history = pd.DataFrame({t: walk(len(idx5), 40 * np.exp(rng.uniform(0, 2.5)))
                            for t in sc.SECTOR_ETFS}, index=idx5)
    holdings = {e: [(s, None) for s in sc.FALLBACK_HOLDINGS[e]] for e in sc.SECTOR_ETFS}
    syms = sorted({s for lst in holdings.values() for s, _ in lst})
    closes = pd.DataFrame({s: walk(252, 30 * np.exp(rng.uniform(0, 3))) for s in syms}, index=idx5[-252:])
    years = pd.to_datetime([f"{y}-12-31" for y in range(today.year - 1, today.year - 8, -1)])

    def series(base, growth, noise):
        return pd.Series([base * (1 + growth) ** -i * (1 + rng.normal(0, noise)) for i in range(len(years))],
                         index=years)

    bundles = {}
    for s in syms:
        rev = series(rng.uniform(20, 300) * 1e9, rng.uniform(-0.02, 0.15), 0.04)
        margin = rng.uniform(0.05, 0.4)
        op = rev * margin * (1 + rng.normal(0, 0.1, len(years)))
        ni = op * 0.78
        fcf = ni * rng.uniform(0.5, 1.3)
        shares = series(rng.uniform(1, 10) * 1e9, rng.uniform(-0.03, 0.02), 0.002)
        price = float(closes[s].iloc[-1])
        rm = rng.uniform(1.4, 3.6)
        info = {"shortName": f"{s} Demo Corp", "marketCap": price * shares.iloc[0], "regularMarketPrice": price,
                "recommendationMean": rm, "numberOfAnalystOpinions": int(rng.integers(10, 50)),
                "recommendationKey": "strong_buy" if rm < 1.5 else "buy" if rm < 2.5 else "hold" if rm < 3.5 else "sell",
                "targetMeanPrice": price * rng.uniform(0.9, 1.35), "revenueGrowth": rng.uniform(-0.05, 0.25),
                "profitMargins": margin * 0.78, "returnOnEquity": rng.uniform(0.02, 0.4),
                "debtToEquity": rng.uniform(10, 200), "freeCashflow": float(fcf.iloc[0])}
        stmts = {"revenue": rev, "op_income": op, "ebit": op, "net_income": ni, "shares": shares,
                 "shares_out": shares, "equity": rev * rng.uniform(0.2, 1.0), "debt": rev * rng.uniform(0, 0.8),
                 "cash": rev * rng.uniform(0.05, 0.4), "fcf": fcf, "ocf": fcf * 1.2,
                 "sbc": rev * rng.uniform(0.005, 0.08), "deferred_rev": None}
        bundles[s] = {"info": info, "stmts": stmts, "sec": None, "stmt_filled": [],
                      "earnings": today + dt.timedelta(days=int(rng.integers(2, 80))),
                      "news": ["🏛️ Demo headline: new export rules weigh on the sector",
                               "📅 Demo headline: earnings preview"]}

    sc.sector_fetch = lambda: (history, {})
    sc.fetch_fed_rate = lambda: {"label": "3.75–4.00%", "kind": "target range", "date": str(today)}
    sc.deep_fetch = lambda etfs, include_stmts=False, sec_contact="": (
        holdings, {}, {e: ("static", None) for e in etfs}, closes, bundles, {})


# ---------------------------------------------------------------- charts
def make_chart(s, sym, path):
    """Same view as the scanner's chart (6 months + 100-day MA), drawn for e-ink,
    plus the 6-month Fibonacci levels the deep scan scores on and a MACD panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from PIL import Image

    s = s.dropna()
    ma = s.rolling(100).mean()
    ema12, ema26 = s.ewm(span=12, adjust=False).mean(), s.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    view = s.index >= s.index[-1] - pd.DateOffset(months=6)
    s6, ma6, l6, sg6, h6 = s[view], ma[view], line[view], signal[view], hist[view]
    hi, lo = float(s6.max()), float(s6.min())

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 5.6), dpi=120, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    for lv in FIB_LEVELS:
        y = lo + lv * (hi - lo)
        ax.axhline(y, color="#9a9a9a", lw=0.8, zorder=1)
        ax.annotate(f"{lv * 100:g}%", xy=(1.01, y), xycoords=("axes fraction", "data"),
                    fontsize=8, va="center", color="#333")
    if ma6.notna().any():
        ax.plot(ma6.index, ma6.values, color="#555", lw=1.8, ls="--", label="100-day MA", zorder=2)
    ax.plot(s6.index, s6.values, color="black", lw=2.2, label="Price", zorder=3)
    ax.set_title(f"{sym}  {s6.iloc[-1]:.2f}   6 months · Fib = % of 6-mo range",
                 fontsize=11, loc="left", fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="#000")
    ax2.bar(h6.index, h6.values, width=1.0, color=["black" if v >= 0 else "#8a8a8a" for v in h6.values])
    ax2.plot(l6.index, l6.values, color="black", lw=1.1)
    ax2.plot(sg6.index, sg6.values, color="#666", lw=1.1, ls="--")
    ax2.set_ylabel("MACD", fontsize=9)
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    Image.open(buf).convert("L").save(path, optimize=True)


# ---------------------------------------------------------------- html
CSS = """
body{font-family:Georgia,serif;color:#000;background:#fff;margin:12px;font-size:18px}
a{color:#000}
h1{font-size:26px;margin:0 0 4px} h2{font-size:21px;margin:18px 0 6px;border-bottom:3px solid #000}
.fed{float:right;font-size:15px;border:2px solid #000;padding:2px 6px;text-align:right}
.stamp{font-size:14px;margin-bottom:6px} .nav{font-size:16px;margin-bottom:8px}
.mood{font-weight:bold;border:3px solid #000;padding:8px;margin:8px 0}
table{width:100%;border-collapse:collapse}
th{border-bottom:3px solid #000;text-align:right;padding:6px 4px}
td{border-bottom:1px solid #000;text-align:right;padding:7px 4px;vertical-align:top}
.l{text-align:left} small{font-size:13px}
.neg{background:#000;color:#fff;padding:1px 4px}
.tag{padding:1px 5px;font-weight:bold;font-size:15px;white-space:nowrap}
.strong{background:#000;color:#fff} .weak{border:2px dashed #000} .na{border:1px solid #888;color:#555}
.card{border:2px solid #000;padding:8px;margin:10px 0}
.row{font-size:15px;margin-top:6px} .warn{font-weight:bold;font-size:15px;margin-top:4px}
.src{font-size:13px;margin-top:6px} .note{font-size:13px;margin-top:12px}
img{width:100%;height:auto;border:1px solid #000}
"""


def page(title, body):
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>')


def card(r, up, link=True):
    name = f'<a href="{up}t/{r["sym"]}.html"><b>{r["sym"]}</b></a>' if link else f"<b>{r['sym']}</b>"
    if r["weight"]:
        size = f" ({r['weight'] * 100:.0f}% of fund)"
    elif r["mcap"]:
        size = f" (mkt cap {sc.fmt_cap(r['mcap'])})"
    else:
        size = ""
    px = f"{r['price']:,.2f} <b>{day_html(r['day_pct'])}</b>" if r["price"] else "price n/a"
    out = [f'<div class="card"><div>{name} <small>{html.escape(r["name"])}{size}</small>'
           f'<span style="float:right">{px}</span></div>']
    if r["warn"]:
        out.append(f'<div class="warn">{text_html(r["warn"])}</div>')
    out.append(f'<div class="row"><b>Compounder</b> {dot_tag(r["comp"])} {text_html(r["comp"]["desc"])}</div>')
    out.append(f'<div class="row"><b>Dip-buyer</b> {dot_tag(r["dip"])} {text_html(r["dip"]["desc"])}</div>')
    if r["sec_line"]:
        out.append(f'<div class="src">{text_html(r["sec_line"])}</div>')
    if r["news"]:
        out.append(f'<div class="src">News: {" | ".join(text_html(n) for n in r["news"])}</div>')
    out.append("</div>")
    return "".join(out)


def greens(d, etf, style):
    rows = [d["stocks"][s][style] for s, _ in d["holdings"].get(etf, []) if s in d["stocks"]]
    return f"{sum(r['dot'] == sc.GREEN_DOT for r in rows)}/{len(rows)}"


def render_index(d, now, note):
    tbl = d["table"].sort_values("Day %", ascending=False)
    rows = []
    for _, x in tbl.iterrows():
        etf, sector = x["Ticker"], SHORT.get(x["Sector"], x["Sector"])
        rows.append(f'<tr><td class="l"><a href="s/{etf}.html"><b>{etf}</b></a><br><small>{html.escape(sector)}'
                    f'<br>C {greens(d, etf, "comp")} · D {greens(d, etf, "dip")} green</small></td>'
                    f'<td>{x["Price"]:,.2f}</td><td><b>{day_html(x["Day %"])}</b></td>'
                    f'<td>{pct(x["3-Mo %"])}</td><td>{pct(x["1-Yr %"])}</td><td>{pct(x["5-Yr %"])}</td></tr>')

    reds_, greens_ = sc.top_movers(d["table"])[1], sc.top_movers(d["table"])[0]
    if greens_ and reds_:
        mood = f"Money is moving out of: {', '.join(reds_)} → into: {', '.join(greens_)}"
    elif greens_:
        mood = "Every sector is green — the market is happy today"
    elif reds_:
        mood = "Every sector is red — the market is sad today"
    else:
        mood = "Market is flat today — no sector moved."

    def best(style):
        picks = sorted((r for r in d["stocks"].values() if r[style]["dot"] == sc.GREEN_DOT),
                       key=lambda r: r[style]["score"], reverse=True)[:5]
        if not picks:
            return '<tr><td class="l">No green dots right now</td></tr>'
        return "".join(f'<tr><td class="l"><a href="t/{r["sym"]}.html"><b>{r["sym"]}</b></a> '
                       f'<small>{html.escape(SHORT.get(sc.SECTOR_ETFS[r["etf"]], sc.SECTOR_ETFS[r["etf"]]))}</small>'
                       f'</td><td>{dot_tag(r[style])}</td></tr>' for r in picks)

    fed = d["fed"]
    fed_html = (f'<span class="fed">Fed funds {html.escape(fed["label"])}<br><small>{html.escape(fed["kind"])}</small></span>'
                if fed else "")
    banner = sc.holdings_banner(d["sources"])
    body = f"""{fed_html}<h1>Sector Board{(' &mdash; ' + note) if note else ''}</h1>
<div class="stamp">Updated {now:%a %b %d, %I:%M %p}</div>
<div class="mood">{html.escape(mood)}</div>
<table><tr><th class="l">Sector</th><th>Price</th><th>Day</th><th>3-Mo</th><th>1-Yr</th><th>5-Yr</th></tr>
{''.join(rows)}</table>
<h2>Best Compounders</h2><table>{best('comp')}</table>
<h2>Best Dip-buys</h2><table>{best('dip')}</table>
{f'<div class="note">{text_html(banner)}</div>' if banner else ''}
<div class="note">Financials: {'SEC EDGAR 10-K/10-Q filings, Yahoo fills gaps' if d['sec_on'] else 'Yahoo (SEC not connected)'}.
Tap a sector for its deep scan, a ticker for its chart. Black box = down today.
GREEN = green dot, RED (dashed) = red dot; the number is the scanner's score.
Screening signals, not buy/sell advice.</div>"""
    return page("Sector Board", body)


def render_sector(d, etf, now):
    x = d["table"][d["table"]["Ticker"] == etf]
    head = (f'{x["Price"].iloc[0]:,.2f} <b>{day_html(x["Day %"].iloc[0])}</b> · ' if len(x) else "")
    kind, date = d["sources"].get(etf, ("live", None))
    src = sc.SOURCE_NOTES[kind].format(date=date or "")
    if etf in d["errors"]:
        cards = f'<div class="card">Top holdings unavailable: {html.escape(str(d["errors"][etf]))}</div>'
    else:
        cards = "".join(card(d["stocks"][s], "../") for s, _ in d["holdings"].get(etf, []) if s in d["stocks"])
    img = f'<img src="../img/{etf}.png" alt="{etf} chart">' if os.path.exists(f"{OUT}/img/{etf}.png") else ""
    body = f"""<div class="nav"><a href="../index.html">&larr; Board</a></div>
<h1>{etf} &middot; {html.escape(sc.SECTOR_ETFS[etf])}</h1>
<div class="stamp">{head}Updated {now:%I:%M %p}</div>{img}
<h2>Deep scan: Compounder {greens(d, etf, 'comp')} green · Dip-buyer {greens(d, etf, 'dip')} green</h2>
<div class="src">{html.escape(src)}</div>{cards}"""
    return page(f"{etf} deep scan", body)


def render_ticker(r, now):
    etf = r["etf"]
    img = f'<img src="../img/{r["sym"]}.png" alt="{r["sym"]} chart">' if os.path.exists(f"{OUT}/img/{r['sym']}.png") else ""
    body = f"""<div class="nav"><a href="../index.html">&larr; Board</a> &nbsp;
<a href="../s/{etf}.html">&larr; {etf} {html.escape(sc.SECTOR_ETFS[etf])}</a></div>
<div class="stamp">Updated {now:%I:%M %p}</div>{img}{card(r, '../', link=False)}"""
    return page(f"{r['sym']} chart", body)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="made-up data, no internet needed")
    args = ap.parse_args()
    if args.demo:
        install_demo()
    d = gather()
    now = dt.datetime.now()
    if d is None:
        # Yahoo sent no sector prices (it sometimes limits cloud servers like GitHub's).
        # Don't publish an empty board: the last good one stays online.
        print("::warning::Yahoo returned no sector prices this run - kept the last published page.")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
                fh.write("publish=false\n")
        return

    for sub in ("s", "t", "img"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    series = {etf: d["history"][etf] for etf in sc.SECTOR_ETFS if etf in d["history"].columns}
    series.update({sym: r["series"] for sym, r in d["stocks"].items() if r["series"] is not None})
    for sym, s in series.items():
        if s.dropna().shape[0] >= 30:
            try:
                make_chart(s, sym, f"{OUT}/img/{sym}.png")
            except Exception as exc:
                print(f"chart {sym} skipped: {exc}")

    def write(path, text):
        with open(os.path.join(OUT, path), "w", encoding="utf-8") as fh:
            fh.write(text)

    write("index.html", render_index(d, now, "DEMO DATA" if args.demo else ""))
    for etf in sc.SECTOR_ETFS:
        write(f"s/{etf}.html", render_sector(d, etf, now))
    for r in d["stocks"].values():
        write(f"t/{r['sym']}.html", render_ticker(r, now))
    print(f"Wrote {OUT}/: {len(sc.SECTOR_ETFS)} sector pages, {len(d['stocks'])} stock pages")


if __name__ == "__main__":
    main()
