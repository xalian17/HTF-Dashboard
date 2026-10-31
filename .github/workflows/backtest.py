#!/usr/bin/env python3
"""
backtest.py — did the dashboard's calls actually work?

    python backtest.py

Replays the live scoring pipeline (fetch.compute_indicators + scoring.aggregate,
current config: EMA, same thresholds) week by week over as much history as the
data allows, then measures what BTC did AFTER each week's call:

  - forward return at +13 / +26 / +52 weeks (median, % of weeks positive)
  - worst drawdown within the next 26 weeks (median) — "was it a bad time to act"

Read-only: changes nothing on the dashboard. Writes backtest.csv (one row per week)
as a workflow artifact and prints summary tables to the log.

PRICE: Bitstamp daily BTC/USD (back to 2011) resampled to Monday-open weeks, so the
200W EMA is valid from ~2015 and the test covers 3 cycles. Falls back to Binance
(2017+, ~1 cycle) if Bitstamp is unreachable. Uses one source for the whole run so
the annual VWAP isn't distorted by a mid-year volume switch.

ON-CHAIN: same BGeometrics series as live (5 calls). Weeks before a series starts
are skipped.

HONEST LIMITS
  - In-sample: thresholds were tuned while looking at this same history, so results
    flatter the system somewhat.
  - Overlapping windows: consecutive weekly signals share most of their forward path.
    The "episodes" count shows how many INDEPENDENT signal runs there really were.
  - MSB/BOS (manual) isn't in history -> scored as none throughout.
"""

import math
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import fetch
import scoring
from backfill import fetch_onchain_series, value_asof

BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
BITSTAMP_START = datetime(2011, 8, 15, tzinfo=timezone.utc)   # a Monday
HORIZONS = (13, 26, 52)
DD_WINDOW = 26

# Valuation buckets for the Cheap/Fair/Expensive x Posture table.
# valuation = value_core + chain_core (the two ungated axes). Provisional cutoffs.
CHEAP_AT = 2.5
EXPENSIVE_AT = -2.5

ERAS = [
    ("2015-2018 cycle", "2015-01-01", "2018-12-31"),
    ("2019-2022 cycle", "2019-01-01", "2022-12-31"),
    ("2023-now cycle",  "2023-01-01", "2100-01-01"),
]


# ----------------------------------------------------------------------
# price
# ----------------------------------------------------------------------

def fetch_bitstamp_weekly():
    now = datetime.now(timezone.utc)
    t = int(BITSTAMP_START.timestamp())
    rows = []
    for _ in range(20):                       # ~5,500 days / 1,000 per page
        r = requests.get(BITSTAMP_URL, params={"step": 86400, "limit": 1000, "start": t}, timeout=60)
        r.raise_for_status()
        page = r.json()["data"]["ohlc"]
        if not page:
            break
        rows.extend(page)
        last = int(page[-1]["timestamp"])
        if len(page) < 1000 or last >= now.timestamp() - 2 * 86400:
            break
        t = last + 86400
        time.sleep(1)
    if not rows:
        raise RuntimeError("Bitstamp returned no data")

    d = pd.DataFrame(rows)
    d["ts"] = pd.to_datetime(d["timestamp"].astype(int), unit="s", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = d[c].astype(float)
    d = d.drop_duplicates("ts").sort_values("ts").set_index("ts")

    w = d.resample("W-MON", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    # fill gap weeks (early Bitstamp) so the weekly grid stays continuous
    w["close"] = w["close"].ffill()
    for c in ("open", "high", "low"):
        w[c] = w[c].fillna(w["close"])
    w["volume"] = w["volume"].fillna(0.0)
    w = w.dropna(subset=["close"])

    out = [{"open_time": ts.to_pydatetime(), "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume} for ts, r in w.iterrows()]
    return out


def get_weekly_prices():
    try:
        kl = fetch_bitstamp_weekly()
        return kl, "Bitstamp BTC/USD (daily -> weekly)"
    except Exception as e:  # noqa: BLE001
        print(f"  Bitstamp failed ({e}) — falling back to Binance (2017+, shorter test)", file=sys.stderr)
        return fetch.fetch_klines(), "Binance BTCUSDT (fallback)"


# ----------------------------------------------------------------------
# replay
# ----------------------------------------------------------------------

def replay(klines, series):
    now = datetime.now(timezone.utc)
    rows = []
    for i, k in enumerate(klines):
        wk_open = k["open_time"]
        if wk_open + timedelta(days=7) > now:      # forming week: no closed read
            break
        if i < fetch.MA_PERIODS["ma200"]:          # 200W EMA not seeded yet
            continue
        ind = fetch.compute_indicators(klines[: i + 1])
        if any(math.isnan(ind[x]) for x in ("dev200", "p35", "p50", "rsi")):
            continue
        last_day = (wk_open + timedelta(days=6)).date()
        oc = {n: value_asof(series[n], last_day) for n in ("mvrvZ", "sth", "lth")}
        if any(v is None for v in oc.values()):
            continue
        s = scoring.aggregate({
            "vwapSigma": ind["vwapSigma"], "rsi": ind["rsi"], "dev200": ind["dev200"],
            "above50W": ind["above50W"], "ma3550": ind["ma3550"], "price": ind["price"],
            "sth": oc["sth"], "lth": oc["lth"], "mvrvZ": oc["mvrvZ"],
            "year": fetch.cycle_year(wk_open), "event": {"type": "none", "weeks": 0},
        })
        rows.append({
            "i": i, "week": wk_open.strftime("%Y-%m-%d"), "close": k["close"],
            "call": s["call"], "composite": s["composite"],
            "value": s["valueCore"], "trend": s["trendGated"], "chain": s["chainCore"],
            "above50W": ind["above50W"], "dev200": ind["dev200"],
            "confidence": s["confidence"]["tier"],
        })
    return pd.DataFrame(rows)


def add_forward(df, klines):
    closes = np.array([k["close"] for k in klines], dtype=float)
    n = len(closes)
    for h in HORIZONS:
        df[f"r{h}"] = [closes[i + h] / closes[i] - 1 if i + h < n else np.nan for i in df["i"]]
    df[f"dd{DD_WINDOW}"] = [
        closes[i + 1: i + 1 + DD_WINDOW].min() / closes[i] - 1 if i + DD_WINDOW < n else np.nan
        for i in df["i"]]
    return df


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------

def episodes(mask):
    """Number of separate runs of consecutive True weeks."""
    m = list(mask)
    return sum(1 for j, v in enumerate(m) if v and (j == 0 or not m[j - 1]))


def stats_line(label, sub, ep=None):
    def med(c):
        v = sub[c].dropna()
        return f"{v.median()*100:+7.1f}%" if len(v) else "     —  "

    def pos(c):
        v = sub[c].dropna()
        return f"{(v > 0).mean()*100:4.0f}%" if len(v) else "   —"

    ep_s = f"{ep:>4d}" if ep is not None else "    "
    return (f"  {label:<34s}{len(sub):>5d} {ep_s}  "
            + "  ".join(f"{med(f'r{h}')} {pos(f'r{h}')}" for h in HORIZONS)
            + f"   {med(f'dd{DD_WINDOW}')}")


def header(title):
    print(f"\n{title}")
    print(f"  {'':<34s}{'weeks':>5s} {'eps':>4s}  "
          + "  ".join(f"{('+' + str(h) + 'w med  %up'):>14s}" for h in HORIZONS)
          + f"   {'26w drawdn':>10s}")


def report(df, source):
    print("=" * 100)
    print(f"BACKTEST — price: {source} | MA: {fetch.MA_TYPE} | "
          f"{df['week'].iloc[0]} -> {df['week'].iloc[-1]} | {len(df)} scored weeks")
    print("Forward return = BTC close N weeks after the week's close. 'eps' = independent signal runs.")
    print("=" * 100)

    order = ["AGGRESSIVE ACCUMULATE", "ACCUMULATE", "NEUTRAL", "DISTRIBUTE", "AGGRESSIVE SELL"]
    header("1) BY CALL")
    print(stats_line("ALL WEEKS (baseline)", df))
    for c in order:
        m = df["call"] == c
        if m.any():
            print(stats_line(c, df[m], episodes(m)))

    header("2) ACCUMULATE-type calls, split by 50W regime  (tests: does 'below 50W' matter?)")
    acc = df["call"].str.contains("ACCUMULATE")
    for lab, m in (("accumulate · ABOVE 50W", acc & df["above50W"]),
                   ("accumulate · BELOW 50W", acc & ~df["above50W"])):
        if m.any():
            print(stats_line(lab, df[m], episodes(m)))

    header("3) VALUATION x POSTURE  (the proposed split; valuation = value + on-chain axes)")
    val = df["value"] + df["chain"]
    vb = np.where(val >= CHEAP_AT, "Cheap", np.where(val <= EXPENSIVE_AT, "Expensive", "Fair"))
    for v in ("Cheap", "Fair", "Expensive"):
        for above, plab in ((True, "above 50W"), (False, "below 50W")):
            m = (vb == v) & (df["above50W"] == above)
            if m.any():
                print(stats_line(f"{v} · {plab}", df[m], episodes(m)))

    header("4) 200W position  (tests: is 'below the 200W' a buy or a breakdown?)")
    for lab, m in (("below 200W", df["dev200"] <= 0),
                   ("0-30% above", (df["dev200"] > 0) & (df["dev200"] <= 30)),
                   ("30-100% above", (df["dev200"] > 30) & (df["dev200"] <= 100)),
                   (">100% above", df["dev200"] > 100)):
        if m.any():
            print(stats_line(lab, df[m], episodes(m)))

    header("5) CONFIDENCE tier on ACCUMULATE-type calls  (tests: does the tier mean anything?)")
    for t in ("High", "Moderate-high", "Moderate", "Low"):
        m = acc & (df["confidence"] == t)
        if m.any():
            print(stats_line(t, df[m], episodes(m)))

    header("6) BY ERA  (does it hold in each cycle, or only one?)")
    for name, a, b in ERAS:
        e = (df["week"] >= a) & (df["week"] <= b)
        if not e.any():
            continue
        print(stats_line(f"{name} · all", df[e]))
        m = e & acc
        if m.any():
            print(stats_line(f"{name} · accumulate", df[m], episodes(m)))
        m = e & df["call"].isin(["DISTRIBUTE", "AGGRESSIVE SELL"])
        if m.any():
            print(stats_line(f"{name} · distribute/sell", df[m], episodes(m)))
    print("\nIn-sample test, overlapping windows — treat as evidence, not proof.")


def main():
    print("Fetching price history…")
    klines, source = get_weekly_prices()
    print(f"  {len(klines)} weekly candles from {klines[0]['open_time'].date()} ({source})")
    print("Fetching on-chain history…")
    series = {}
    for name in ("mvrvZ", "sth", "lth"):
        try:
            series[name] = fetch_onchain_series(name, fetch.ENDPOINTS[name])
            print(f"  {name}: {series[name][0][0]} -> {series[name][-1][0]}")
        except Exception as e:  # noqa: BLE001
            sys.exit(f"On-chain '{name}' unavailable ({e}). BGeometrics free tier is ~8 req/hr — "
                     f"wait an hour and rerun.")
    print("Replaying weekly scores… (takes a minute)")
    df = replay(klines, series)
    if df.empty:
        sys.exit("No weeks could be scored — check the data ranges above.")
    df = add_forward(df, klines)
    df.drop(columns=["i"]).to_csv("backtest.csv", index=False)
    report(df, source)


if __name__ == "__main__":
    main()
