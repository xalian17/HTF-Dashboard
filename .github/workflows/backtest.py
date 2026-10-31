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

ON-CHAIN: same BGeometrics series as live (3 calls). The free tier only returns
~4 years (late 2022 on), so the report is split:
  PART A  full model (needs on-chain)     late 2022 -> now
  PART B  price-only pieces (200W, 50W, VWAP value axis)   ~2015 -> now

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
    """Daily candles, paged BACKWARDS from now with `end` (1,000 days per call)."""
    end = int(datetime.now(timezone.utc).timestamp())
    stop = int(BITSTAMP_START.timestamp())
    rows = []
    for _ in range(12):                       # ~5,500 days / 1,000 per page
        r = requests.get(BITSTAMP_URL, params={"step": 86400, "limit": 1000, "end": end}, timeout=60)
        r.raise_for_status()
        page = r.json()["data"]["ohlc"]
        if not page:
            break
        rows.extend(page)
        first = min(int(p["timestamp"]) for p in page)
        if first <= stop or first >= end:     # reached 2011, or no progress
            break
        end = first - 86400
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
    # sanity: must reach (nearly) today and go back before 2017, else it's no better than Binance
    age = (datetime.now(timezone.utc) - out[-1]["open_time"]).days
    if age > 21 or out[0]["open_time"].year >= 2017:
        raise RuntimeError(f"incomplete history ({out[0]['open_time'].date()} -> "
                           f"{out[-1]['open_time'].date()}, {len(out)} weeks)")
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
    """One row per closed week once the 200W EMA exists.
    Price-only fields (value axis, trend, 50W, 200W) for every week; the full
    composite/call only for weeks where all three on-chain series have data."""
    now = datetime.now(timezone.utc)
    rows = []
    for i, k in enumerate(klines):
        wk_open = k["open_time"]
        if wk_open + timedelta(days=7) > now:      # forming week: no closed read
            break
        if i < fetch.MA_PERIODS["ma200"]:          # 200W EMA not seeded yet
            continue
        ind = fetch.compute_indicators(klines[: i + 1])
        if any(math.isnan(ind[x]) for x in ("dev200", "p35", "p50", "rsi", "vwapSigma")):
            continue
        value_px = scoring.score_vwap(ind["vwapSigma"])["score"] + scoring.score_200w(ind["dev200"])["score"]
        row = {
            "i": i, "week": wk_open.strftime("%Y-%m-%d"), "close": k["close"],
            "above50W": ind["above50W"], "dev200": ind["dev200"], "rsi": ind["rsi"],
            "value": value_px, "full": False,
            "call": None, "composite": np.nan, "trend": np.nan, "chain": np.nan, "confidence": None,
            "valScore": np.nan,
        }
        last_day = (wk_open + timedelta(days=6)).date()
        oc = {n: value_asof(series[n], last_day) for n in ("mvrvZ", "sth", "lth")}
        if all(v is not None for v in oc.values()):
            sc = scoring.aggregate({
                "vwapSigma": ind["vwapSigma"], "rsi": ind["rsi"], "dev200": ind["dev200"],
                "above50W": ind["above50W"], "ma3550": ind["ma3550"], "price": ind["price"],
                "sth": oc["sth"], "lth": oc["lth"], "mvrvZ": oc["mvrvZ"],
                "year": fetch.cycle_year(wk_open), "event": {"type": "none", "weeks": 0},
            })
            row.update(full=True, call=sc["call"], composite=sc["composite"], trend=sc["trendGated"],
                       chain=sc["chainCore"], confidence=sc["confidence"]["tier"],
                       valScore=sc["valScore"])
        rows.append(row)
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
    return (f"  {label:<36s}{len(sub):>5d} {ep_s}  "
            + "  ".join(f"{med(f'r{h}')} {pos(f'r{h}')}" for h in HORIZONS)
            + f"   {med(f'dd{DD_WINDOW}')}")


def header(title):
    print(f"\n{title}")
    print(f"  {'':<36s}{'weeks':>5s} {'eps':>4s}  "
          + "  ".join(f"{('+' + str(h) + 'w med  %up'):>14s}" for h in HORIZONS)
          + f"   {'26w drawdn':>10s}")


def report(all_df, source):
    df = all_df[all_df["full"]].copy()
    print("=" * 108)
    print(f"BACKTEST — price: {source} | MA: {fetch.MA_TYPE}")
    print("Forward return = BTC close N weeks after the week's close. 'eps' = independent signal runs.")
    print("=" * 108)

    # ---------------- PART A: full model ----------------
    if df.empty:
        print("\nPART A skipped — no week had both price and on-chain data.")
    else:
        print(f"\nPART A — FULL MODEL (needs on-chain): {df['week'].iloc[0]} -> {df['week'].iloc[-1]}, "
              f"{len(df)} weeks. Limited by the on-chain source's free-tier history.")
        order = ["AGGRESSIVE ACCUMULATE", "ACCUMULATE", "NEUTRAL", "DISTRIBUTE", "AGGRESSIVE SELL"]
        header("A1) BY CALL")
        print(stats_line("ALL WEEKS (baseline)", df))
        for c in order:
            m = df["call"] == c
            if m.any():
                print(stats_line(c, df[m], episodes(m)))

        acc = df["call"].str.contains("ACCUMULATE")
        header("A2) ACCUMULATE-type calls by 50W regime  (does 'below 50W' matter?)")
        for lab, m in (("accumulate · ABOVE 50W", acc & df["above50W"]),
                       ("accumulate · BELOW 50W", acc & ~df["above50W"])):
            if m.any():
                print(stats_line(lab, df[m], episodes(m)))

        header("A3) VALUATION x POSTURE  (proposed split; valuation = value + on-chain axes)")
        val = df["value"] + df["chain"]
        vb = np.where(val >= CHEAP_AT, "Cheap", np.where(val <= EXPENSIVE_AT, "Expensive", "Fair"))
        for v in ("Cheap", "Fair", "Expensive"):
            for above, plab in ((True, "above 50W"), (False, "below 50W")):
                m = (vb == v) & (df["above50W"] == above)
                if m.any():
                    print(stats_line(f"{v} · {plab}", df[m], episodes(m)))

        header("A4) CONFIDENCE tier on ACCUMULATE-type calls  (does the tier mean anything?)")
        for t in ("High", "Moderate-high", "Moderate", "Low"):
            m = acc & (df["confidence"] == t)
            if m.any():
                print(stats_line(t, df[m], episodes(m)))

    # ---------------- PART B: price-only, long history ----------------
    p = all_df
    print(f"\nPART B — PRICE-ONLY PIECES (no on-chain needed): {p['week'].iloc[0]} -> {p['week'].iloc[-1]}, "
          f"{len(p)} weeks")
    header("B1) 200W position  (is 'below the 200W' a buy or a breakdown?)")
    print(stats_line("ALL WEEKS (baseline)", p))
    for lab, m in (("below 200W", p["dev200"] <= 0),
                   ("0-30% above", (p["dev200"] > 0) & (p["dev200"] <= 30)),
                   ("30-100% above", (p["dev200"] > 30) & (p["dev200"] <= 100)),
                   ("100-200% above", (p["dev200"] > 100) & (p["dev200"] <= 200)),
                   (">200% above", p["dev200"] > 200)):
        if m.any():
            print(stats_line(lab, p[m], episodes(m)))

    header("B2) VALUE axis (VWAP + 200W) by 50W regime  (cheap + below 50W: early or wrong?)")
    for lab, m in (("value >= +1 · above 50W", (p["value"] >= 1) & p["above50W"]),
                   ("value >= +1 · below 50W", (p["value"] >= 1) & ~p["above50W"]),
                   ("value 0 · above 50W", (p["value"] == 0) & p["above50W"]),
                   ("value 0 · below 50W", (p["value"] == 0) & ~p["above50W"]),
                   ("value <= -1 (extended)", p["value"] <= -1)):
        if m.any():
            print(stats_line(lab, p[m], episodes(m)))

    header("B3) BY ERA · value >= +1 vs all  (does price-side value hold in every cycle?)")
    for name, a, b in ERAS:
        e = (p["week"] >= a) & (p["week"] <= b)
        if not e.any():
            continue
        print(stats_line(f"{name} · all", p[e]))
        m = e & (p["value"] >= 1)
        if m.any():
            print(stats_line(f"{name} · value >= +1", p[m], episodes(m)))
    # ---------------- PART C: line in the sand ----------------
    print("\nPART C — LINE IN THE SAND: results for every week AT OR BEYOND each threshold.")
    print("Look for where returns jump / drawdowns shrink (buy side) or returns turn negative /")
    print("drawdowns deepen (sell side) — AND the episode count is still more than a couple.")
    if not df.empty:
        scan(df, "valScore", [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7], ">=",
             "C1) VALUATION score >= X  — BUY side (Nov 2022 -> now, one cycle)")
        scan(df, "valScore", [-1, -2, -2.5, -3, -4, -5, -6], "<=",
             "C2) VALUATION score <= X  — SELL side (Nov 2022 -> now; few expensive weeks in this window)")
        scan(df, "composite", [3.5, 4, 5, 6, 7, 8], ">=",
             "C3) old COMPOSITE >= X  — buy side, for comparison")
        scan(df, "composite", [-1, -2, -3, -4, -5, -7], "<=",
             "C4) old COMPOSITE <= X  — sell side, for comparison")
    scan(p, "value", [1, 2, 3, 4], ">=",
         "C5) PRICE-ONLY value score (VWAP + 200W) >= X  — buy side, 2015 -> now, 3 cycles")
    scan(p, "value", [-1, -2, -3, -4], "<=",
         "C6) PRICE-ONLY value score (VWAP + 200W) <= X  — sell side, 2015 -> now (includes 2017 & 2021 tops)")
    print("\nIn-sample test, overlapping windows — treat as evidence, not proof.")


def scan(d, col, thresholds, op, title):
    header(title)
    print(stats_line("ALL WEEKS (baseline)", d))
    v = d[col]
    for x in thresholds:
        m = (v >= x) if op == ">=" else (v <= x)
        m = m.fillna(False)
        if m.any():
            print(stats_line(f"{col} {op} {x:+g}", d[m], episodes(m)))
        else:
            print(f"  {col + ' ' + op + ' ' + format(x, '+g'):<36s}    0      (never happened)")


def main():
    sys.stdout.reconfigure(line_buffering=True)   # keep log lines in order
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
    print(f"  {len(df)} weeks scored ({int(df['full'].sum()) if not df.empty else 0} with on-chain)")
    if df.empty:
        sys.exit("No weeks could be scored — check the data ranges above.")
    df = add_forward(df, klines)
    df.drop(columns=["i"]).to_csv("backtest.csv", index=False)
    report(df, source)


if __name__ == "__main__":
    main()
