#!/usr/bin/env python3
"""
backfill.py — rebuild history.json for every closed week since the dashboard went live.

    python backfill.py                      # weeks from START_WEEK through the last closed week
    python backfill.py --start 2026-06-22   # pick a different first week (a Monday)

Why this exists: the update workflow never committed history.json, so the strip
never accumulated. This replays the SAME pipeline fetch.py runs live, one week at
a time, using only data that existed at that week:

  - price side: weekly klines sliced so week W is the last candle -> exactly what a
    late-in-week live run would have seen (35W/50W/200W, RSI, VWAP, z, reclaim).
  - on-chain: the latest value dated on or before the last day of week W.
  - scoring: scoring.aggregate(), unchanged.

Uses the CURRENT config (MA_TYPE etc.), so the whole strip is on one consistent basis.

Known limitation: MSB/BOS is a manual mark (manual.json) with no saved history, so
every backfilled week is scored with event = none. If you know a week had a live
MSB/BOS, add it to KNOWN_EVENTS below.

Network cost: 1 Binance call + 5 BGeometrics calls (full series each), same as one
normal run. The live forming week is NOT written here; fetch.py adds it on its next run.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

import requests

import fetch
import scoring

START_WEEK = "2026-06-22"   # Monday of the week the dashboard went live (first data.json: Jun 24 2026)

# Optional: {"2026-08-10": {"type": "MSB", "weeks": 0}} — week-open date -> event that was live
KNOWN_EVENTS = {}

DATE_KEYS = ("d", "date", "day", "t", "time", "timestamp", "unixTs", "unix")


def _parse_date(v):
    """BGeometrics dates: 'YYYY-MM-DD', ISO strings, or unix seconds/ms."""
    if v is None:
        return None
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
        x = float(v)
        if x > 1e11:  # ms
            x /= 1000
        return datetime.fromtimestamp(x, tz=timezone.utc).date()
    s = str(v)[:10]
    return datetime.strptime(s, "%Y-%m-%d").date()


def fetch_onchain_series(name, cfg):
    """Full series for one metric -> sorted list of (date, value)."""
    url = f"{fetch.ONCHAIN_BASE}/{cfg['slug']}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError(f"{name}: expected a list of records, got {type(data).__name__}")
    out = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        d = None
        for dk in ("d", "date", "day", "t", "time", "timestamp"):
            if dk in rec:
                d = _parse_date(rec[dk])
                break
        if d is None and "unixTs" in rec:
            d = _parse_date(rec["unixTs"])
        if d is None:
            continue
        val = None
        if cfg.get("value_key") and cfg["value_key"] in rec:
            val = rec[cfg["value_key"]]
        else:
            for k, v in rec.items():
                if k in DATE_KEYS:
                    continue
                try:
                    float(v)
                    val = v
                    break
                except (TypeError, ValueError):
                    continue
        try:
            out.append((d, float(val)))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda t: t[0])
    if not out:
        raise ValueError(f"{name}: no usable records")
    return out


def value_asof(series, day):
    """Latest value dated on or before `day` (None if the series starts later)."""
    best = None
    for d, v in series:
        if d <= day:
            best = v
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START_WEEK, help="first week to rebuild (Monday, YYYY-MM-DD)")
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"Backfill — MA_TYPE={fetch.MA_TYPE}, from week of {start.date()}")
    klines = fetch.fetch_klines()
    print(f"  price: {len(klines)} weekly candles")

    series = {}
    for name, cfg in fetch.ENDPOINTS.items():
        try:
            series[name] = fetch_onchain_series(name, cfg)
            print(f"  on-chain {name}: {len(series[name])} points, "
                  f"{series[name][0][0]} -> {series[name][-1][0]}")
        except Exception as e:  # noqa: BLE001
            print(f"  on-chain {name}: FAILED ({e})", file=sys.stderr)
            series[name] = []
    for req in ("mvrvZ", "sth", "lth"):
        if not series[req]:
            sys.exit(f"Required on-chain series '{req}' unavailable — cannot score. Try again later "
                     f"(BGeometrics free tier is ~8 req/hr).")

    now = datetime.now(timezone.utc)
    hist = []
    for i, k in enumerate(klines):
        wk_open = k["open_time"]
        if wk_open < start:
            continue
        if wk_open + timedelta(days=7) > now:   # still forming -> leave for fetch.py
            break
        ind = fetch.compute_indicators(klines[: i + 1])
        last_day = (wk_open + timedelta(days=6)).date()
        onchain = {n: value_asof(series[n], last_day) for n in series}
        if any(onchain[r] is None for r in ("mvrvZ", "sth", "lth")):
            print(f"  skip {ind['week']}: on-chain data not available for that date")
            continue
        event = KNOWN_EVENTS.get(wk_open.strftime("%Y-%m-%d"), {"type": "none", "weeks": 0})
        score_input = {
            "vwapSigma": ind["vwapSigma"], "rsi": ind["rsi"], "dev200": ind["dev200"],
            "above50W": ind["above50W"], "ma3550": ind["ma3550"], "price": ind["price"],
            "sth": onchain["sth"], "lth": onchain["lth"], "mvrvZ": onchain["mvrvZ"],
            "year": fetch.cycle_year(wk_open), "event": event,
        }
        s = scoring.aggregate(score_input)
        hist.append({
            "week": ind["week"],
            "date": (wk_open + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "composite": round(s["composite"], 2),
            "call": s["call"],
            "phase": s["phase"],
            "valuation": s["valuation"],
            "valScore": s["valScore"],
            "posture": s["posture"],
            "backfilled": True,
        })
        print(f"  {ind['week']}: {s['valuation']:<9s} {s['valScore']:+5.1f} · {s['posture']:<9s} | "
              f"{s['call']:<14s} {s['composite']:+.2f}")

    if not hist:
        sys.exit("No closed weeks in range — nothing written.")
    hist = hist[-fetch.HISTORY_MAX:]
    with open(fetch.HISTORY_PATH, "w") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)
    print(f"Wrote {fetch.HISTORY_PATH}: {len(hist)} closed weeks "
          f"({hist[0]['week']} -> {hist[-1]['week']}). The live week is added by the next fetch.py run.")


if __name__ == "__main__":
    main()
