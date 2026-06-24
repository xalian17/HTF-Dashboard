#!/usr/bin/env python3
"""
calibrate.py — threshold SANITY CHECK for the BTC Weekly Cycle Compass.

This is a DIAGNOSTIC, not an optimizer. It pulls the full available history of each
scored metric from the SAME sources the dashboard uses (Binance + BGeometrics) and
prints, for each:
    - the distribution (min / p5 / p25 / median / p75 / p95 / max)
    - how often each scoring band actually fires (% of weeks in each score bucket)

USE IT TO ANSWER: do the bands realistically fire? e.g. does STH premium exceed +40%?
does MVRV still reach 5 in recent cycles? is VWAP +3sigma ever hit? does the 200W
-2 band (>=200%) ever trigger in the modern era?

IT DOES NOT CHANGE ANY THRESHOLD. You read the output and decide — by reasoning, not by
fitting to tops/bottoms — whether a band looks dead (never fires) or miscalibrated
(fires constantly). Any change after that is a separate, deliberate edit to scoring.py.

    python calibrate.py            # full report
    python calibrate.py --recent   # also print a 2022+ ("modern era") slice

Reuses fetch.py (data sources) and scoring.py (the exact band functions), so the
buckets reported here are the buckets the live dashboard uses.
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd
import requests

import fetch
import scoring


# ----------------------------------------------------------------------
def report(name, series, score_fn, recent_cutoff=None, dates=None):
    """Print distribution + band-firing frequency for one metric."""
    s = pd.Series(series, dtype="float64").dropna()
    if len(s) == 0:
        print(f"\n=== {name} ===  (no data)")
        return
    q = s.quantile([0, .05, .25, .5, .75, .95, 1.0]).values
    print(f"\n=== {name} ===  (n={len(s)})")
    print("  min {:+.2f} | p5 {:+.2f} | p25 {:+.2f} | median {:+.2f} | p75 {:+.2f} | p95 {:+.2f} | max {:+.2f}"
          .format(*q))
    buckets = Counter(score_fn(v) for v in s)
    total = sum(buckets.values())
    print("  band firing (how often each score fires):")
    for sc in sorted(buckets, reverse=True):
        bar = "#" * int(round(40 * buckets[sc] / total))
        print(f"    score {sc:+d}: {buckets[sc]:4d}  ({100*buckets[sc]/total:5.1f}%)  {bar}")

    if recent_cutoff is not None and dates is not None:
        d = pd.to_datetime(pd.Series(dates))
        mask = (d >= pd.Timestamp(recent_cutoff)).values
        sr = s[mask[:len(s)]] if len(mask) >= len(s) else s
        sr = pd.Series(sr, dtype="float64").dropna()
        if len(sr):
            rb = Counter(score_fn(v) for v in sr)
            rt = sum(rb.values())
            extreme = sum(v for k, v in rb.items() if abs(k) == 2)
            print(f"  [{recent_cutoff}+ slice, n={len(sr)}] extreme (+/-2) bands fire "
                  f"{100*extreme/rt:.1f}% of the time")


# ----------------------------------------------------------------------
def price_frame():
    kl = fetch.fetch_klines()
    df = pd.DataFrame(kl).sort_values("open_time").reset_index(drop=True)
    return df


def annual_vwap_sigma_series(df):
    """Re-anchored-each-Jan-1 annual VWAP sigma, computed for every weekly bar."""
    out = []
    for i in range(len(df)):
        yr0 = pd.Timestamp(year=int(df["open_time"].iloc[i].year), month=1, day=1, tz="UTC")
        sub = df.iloc[:i + 1]
        sub = sub[sub["open_time"] >= yr0]
        if len(sub) < 2:
            out.append(np.nan)
            continue
        tp = (sub["high"] + sub["low"] + sub["close"]) / 3
        vol = sub["volume"].replace(0, np.nan).ffill().fillna(1.0)
        cv = vol.cumsum()
        vw = (tp * vol).cumsum() / cv
        var = (tp * tp * vol).cumsum() / cv - vw ** 2
        std = np.sqrt(var.clip(lower=0))
        p = sub["close"].iloc[-1]
        out.append((p - vw.iloc[-1]) / std.iloc[-1] if std.iloc[-1] > 0 else 0.0)
    return out


def daily_price_series():
    """Daily closes for accurate date-alignment with on-chain series."""
    url = fetch.BINANCE_BASE + "/api/v3/klines"
    r = requests.get(url, params={"symbol": fetch.SYMBOL, "interval": "1d", "limit": 1000}, timeout=60)
    r.raise_for_status()
    rows = r.json()
    idx = pd.to_datetime([k[0] for k in rows], unit="ms")
    return pd.Series([float(k[4]) for k in rows], index=idx).sort_index()


def fetch_onchain_series(slug):
    """Full {date -> value} history for a BGeometrics metric."""
    r = requests.get(f"{fetch.ONCHAIN_BASE}/{slug}", timeout=60)
    r.raise_for_status()
    data = r.json()
    out = {}
    for rec in (data if isinstance(data, list) else [data]):
        if not isinstance(rec, dict):
            continue
        date = None
        for dk in ("d", "date", "day", "t", "time", "timestamp"):
            if dk in rec:
                date = str(rec[dk])[:10]
                break
        val = None
        for k, v in rec.items():
            if k in ("d", "date", "day", "t", "time", "timestamp", "unixTs", "unix"):
                continue
            try:
                val = float(v)
                break
            except (TypeError, ValueError):
                continue
        if date and val is not None:
            out[date] = val
    return out


def premium_series(cost_basis_by_date, daily_px):
    """Align price to each cost-basis date (asof) and return premium % + dates."""
    cb = pd.Series(cost_basis_by_date).dropna()
    cb.index = pd.to_datetime(cb.index)
    cb = cb.sort_index()
    px = daily_px.reindex(daily_px.index.union(cb.index)).ffill().reindex(cb.index)
    prem = (px / cb - 1) * 100
    return prem.values, cb.index


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", action="store_true", help="also report a 2022+ slice")
    args = ap.parse_args()
    cutoff = "2022-01-01" if args.recent else None

    print("BTC Weekly Cycle Compass — threshold sanity check")
    print("(reports distributions only; does NOT change any threshold)")

    # ---- price-derived metrics ----
    try:
        df = price_frame()
        dates = df["open_time"]
        ma200 = df["close"].rolling(200).mean()
        dev200 = ((df["close"] / ma200 - 1) * 100).values
        report("200W distance %", dev200, lambda v: scoring.score_200w(v)["score"], cutoff, dates)

        delta = df["close"].diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        ag = gain.ewm(alpha=1 / 14, adjust=False).mean()
        al = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rsi = (100 - 100 / (1 + ag / al.replace(0, np.nan))).values
        report("W1 RSI", rsi, lambda v: scoring.score_rsi(v)["score"], cutoff, dates)

        sig = annual_vwap_sigma_series(df)
        report("Annual VWAP sigma", sig, lambda v: scoring.score_vwap(v)["score"], cutoff, dates)
    except Exception as e:  # noqa: BLE001
        print(f"\n[price metrics] FAILED: {e}")

    # ---- on-chain metrics ----
    try:
        mvrv = fetch_onchain_series(fetch.ENDPOINTS["mvrvZ"]["slug"])
        report("MVRV Z-Score", list(mvrv.values()),
               lambda v: scoring.score_mvrv_z(v)["score"], cutoff, list(mvrv.keys()))
    except Exception as e:  # noqa: BLE001
        print(f"\n[MVRV] FAILED: {e}")

    try:
        daily_px = daily_price_series()
        sth = fetch_onchain_series(fetch.ENDPOINTS["sth"]["slug"])
        lth = fetch_onchain_series(fetch.ENDPOINTS["lth"]["slug"])
        sp, sd = premium_series(sth, daily_px)
        report("STH premium %", sp, lambda v: scoring.score_sth(1 + v / 100, 1)["score"], cutoff, sd)
        lp, ld = premium_series(lth, daily_px)
        report("LTH premium %", lp, lambda v: scoring.score_lth(1 + v / 100, 1)["score"], cutoff, ld)
    except Exception as e:  # noqa: BLE001
        print(f"\n[STH/LTH premiums] FAILED: {e}")

    print("\nDone. Read each metric's band-firing %. A band at ~0% never fires (dead);")
    print("a band near 100% fires constantly (no resolution). Decide adjustments by")
    print("reasoning — do NOT fit thresholds to past tops/bottoms.")


if __name__ == "__main__":
    main()
