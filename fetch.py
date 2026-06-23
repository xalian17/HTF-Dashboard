#!/usr/bin/env python3
"""
fetch.py — BTC Weekly Cycle Compass data builder.

Pulls live data, computes the weekly indicators, runs the scoring engine
(scoring.py), and writes data.json for the dashboard to render.

    python fetch.py            # normal run: fetch live, compute, write data.json
    python fetch.py --probe    # hit each endpoint and dump what came back (calibration)
    python fetch.py --demo     # write data.json from built-in mock values (no network)

WHY --probe EXISTS
------------------
The on-chain endpoint slugs and JSON key names below are the best reconstruction
from BGeometrics' documentation, but they were not live-tested in the environment
that authored this file. The FIRST thing to do on a real machine is:

    python fetch.py --probe

It prints the URL it called, the HTTP status, and the last record returned for each
metric. If a slug is wrong you'll see a 404 and the fix is a one-line edit to ENDPOINTS.
The price side (Binance) uses a stable, well-known public endpoint.
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import requests

import scoring

# ======================================================================
# CONFIG  — everything tweakable lives here. None of it is locked.
# ======================================================================

# ---- price source (Binance public klines) ----
# Canonical weekly BTC candle. data-api.binance.vision is the public market-data
# mirror and tends to dodge the geoblocking that can hit api.binance.com from CI.
BINANCE_BASE   = "https://data-api.binance.vision"
BINANCE_FALLBACK = "https://api.binance.com"
SYMBOL         = "BTCUSDT"
INTERVAL       = "1w"
KLINES_LIMIT   = 1000          # ~450 weekly candles exist since 2017; 1000 is the max

# ---- on-chain source (BGeometrics / bitcoin-data.com, free tier) ----
# Free tier: ~8 req/hr, 15/day without a token — a once-daily pull of these is ~5 req.
# VERIFY these slugs + value keys with `--probe`. value_key=None lets the tolerant
# parser auto-pick the non-date field, so a wrong key name often still works.
ONCHAIN_BASE = "https://bitcoin-data.com/v1"
ENDPOINTS = {
    "mvrvZ":         {"slug": "mvrv-zscore",        "value_key": None},
    "sth":           {"slug": "sth-realized-price", "value_key": None},
    "lth":           {"slug": "lth-realized-price", "value_key": None},
    "nupl":          {"slug": "nupl",               "value_key": None},
    "avgBuyPrice":   {"slug": "realized-price",     "value_key": None},
}

# ---- cycle definition (drives year-in-cycle label only; VWAP is annual, see below) ----
# Year is counted from the cycle-low YEAR: bottom year = 1 ... top year = 4,
# matching "1st year bottom / 4th year top" (2018 low -> 2021 top, 2022 low -> 2025 top).
# Update this list when a new cycle bottom confirms.
CYCLE_LOW_YEARS = [2018, 2022]
# VWAP re-anchors every Jan 1 of the current year (annual reset). No cycle anchor needed.

# ---- moving averages ----
MA_TYPE   = "SMA"              # "SMA" | "EMA" | "WMA". Default SMA (canonical 200W).
                              # NOTE: your MA study favored WMA on the 9/20 — switch here if wanted.
MA_PERIODS = {"ma35": 35, "ma50": 50, "ma200": 200}

# ---- z-score window for the 35/50 distance bands ----
Z_LOOKBACK = 200              # trailing weeks for mean/sd of % distance

# ---- RSI ----
RSI_PERIOD = 14

OUT_PATH = "data.json"


# ======================================================================
# Fetch
# ======================================================================

def fetch_klines():
    """Weekly OHLCV from Binance. Returns list of dicts, oldest first."""
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": KLINES_LIMIT}
    last_err = None
    for base in (BINANCE_BASE, BINANCE_FALLBACK):
        url = base + "/api/v3/klines"
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            rows = r.json()
            out = []
            for k in rows:
                out.append({
                    "open_time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                    "open":  float(k[1]),
                    "high":  float(k[2]),
                    "low":   float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            if not out:
                raise ValueError("klines returned empty")
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  [price] {base} failed: {e}", file=sys.stderr)
    raise RuntimeError(f"Could not fetch klines from any Binance host: {last_err}")


def _extract_latest(payload, value_key):
    """Tolerant parser for a BGeometrics-style metric response.
    Accepts a list of records or a single record; returns (date_str, float_value).
    """
    rec = payload[-1] if isinstance(payload, list) else payload
    if not isinstance(rec, dict):
        # bare value or list of values
        return None, float(rec)
    date_str = None
    for dk in ("d", "date", "day", "t", "time", "timestamp"):
        if dk in rec:
            date_str = str(rec[dk])
            break
    if value_key and value_key in rec:
        return date_str, float(rec[value_key])
    # auto-pick: first numeric-looking field that isn't a date/timestamp
    for k, val in rec.items():
        if k in ("d", "date", "day", "t", "time", "timestamp", "unixTs", "unix"):
            continue
        try:
            return date_str, float(val)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"could not find a numeric value in record: {rec}")


def fetch_onchain_metric(name, cfg, probe=False):
    url = f"{ONCHAIN_BASE}/{cfg['slug']}"
    r = requests.get(url, timeout=30)
    if probe:
        sample = None
        try:
            data = r.json()
            sample = (data[-1] if isinstance(data, list) else data)
        except Exception:  # noqa: BLE001
            sample = r.text[:200]
        print(f"  [{name:13s}] {url}\n     status={r.status_code}  last={sample}")
        if r.status_code != 200:
            return None
    r.raise_for_status()
    _, value = _extract_latest(r.json(), cfg["value_key"])
    return value


def fetch_onchain(probe=False):
    out = {}
    for name, cfg in ENDPOINTS.items():
        try:
            out[name] = fetch_onchain_metric(name, cfg, probe=probe)
        except Exception as e:  # noqa: BLE001
            print(f"  [onchain] '{name}' ({cfg['slug']}) failed: {e}", file=sys.stderr)
            out[name] = None
    return out


# ======================================================================
# Indicators  (pandas)
# ======================================================================

def compute_indicators(klines):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(klines).sort_values("open_time").reset_index(drop=True)
    close = df["close"]

    # --- moving averages ---
    def moving_avg(series, period):
        if MA_TYPE == "EMA":
            return series.ewm(span=period, adjust=False).mean()
        if MA_TYPE == "WMA":
            w = np.arange(1, period + 1)
            return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
        return series.rolling(period).mean()  # SMA

    ma35  = moving_avg(close, MA_PERIODS["ma35"])
    ma50  = moving_avg(close, MA_PERIODS["ma50"])
    ma200 = moving_avg(close, MA_PERIODS["ma200"])

    # --- Wilder RSI (matches TradingView at the latest bar) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # --- cycle-anchored VWAP + volume-weighted sigma ---
    now = df["open_time"].iloc[-1]
    # Annual VWAP — re-anchors every Jan 1 (matches the TradingView setup).
    anchor = pd.Timestamp(year=int(now.year), month=1, day=1, tz="UTC")
    sub = df[df["open_time"] >= anchor]
    tp = (sub["high"] + sub["low"] + sub["close"]) / 3
    vol = sub["volume"].replace(0, np.nan).ffill().fillna(1.0)
    cum_v = vol.cumsum()
    vwap = (tp * vol).cumsum() / cum_v
    var = (tp * tp * vol).cumsum() / cum_v - vwap ** 2
    std = np.sqrt(var.clip(lower=0))
    price = float(close.iloc[-1])
    last_vwap = float(vwap.iloc[-1])
    last_std = float(std.iloc[-1])
    vwap_sigma = (price - last_vwap) / last_std if last_std > 0 else 0.0

    # --- % distance + z-scores (35/50) ---
    def pct_and_z(ma_series):
        pct_series = (close / ma_series - 1) * 100
        pct_now = float(pct_series.iloc[-1])
        window = pct_series.dropna().tail(Z_LOOKBACK)
        mu = float(window.mean())
        sd = float(window.std())
        z = (pct_now - mu) / sd if sd > 0 else 0.0
        return pct_now, z

    p35, z35 = pct_and_z(ma35)
    p50, z50 = pct_and_z(ma50)
    dev200 = float((close.iloc[-1] / ma200.iloc[-1] - 1) * 100)

    # --- regime + 35-reclaim band state ---
    ma35_now, ma50_now = float(ma35.iloc[-1]), float(ma50.iloc[-1])
    above50 = price > ma50_now

    if price < ma50_now:
        pos = "below"
    elif price < ma35_now:
        pos = "watching"
    else:
        pos = "above"
    # confirmed reclaim on the last CLOSED weekly candle (index -2; -1 is forming)
    reclaimed = False
    try:
        c_last, c_prev = float(close.iloc[-2]), float(close.iloc[-3])
        m35_last, m35_prev = float(ma35.iloc[-2]), float(ma35.iloc[-3])
        m50_last = float(ma50.iloc[-2])
        reclaimed = (c_last > m35_last) and (c_prev <= m35_prev) and (c_last > m50_last)
    except Exception:  # noqa: BLE001
        pass
    ma3550 = "reclaim" if reclaimed else pos

    # --- day within the current (forming) weekly candle ---
    day_of_week = int((datetime.now(timezone.utc) - now).days) + 1
    day_of_week = max(1, min(7, day_of_week))

    return {
        "price": price,
        "week": now.strftime("%b %d %Y"),
        "dayOfWeek": day_of_week,
        "vwapSigma": round(vwap_sigma, 3),
        "rsi": round(float(rsi.iloc[-1]), 1),
        "dev200": round(dev200, 1),
        "above50W": bool(above50),
        "ma3550": ma3550,
        "p35": round(p35, 1), "z35": round(z35, 2),
        "p50": round(p50, 1), "z50": round(z50, 2),
        "_anchor": anchor.strftime("%Y-%m-%d"),
    }


def cycle_year(now):
    y = now.year
    past = [c for c in CYCLE_LOW_YEARS if c <= y]
    base = max(past) if past else min(CYCLE_LOW_YEARS)
    return y - base + 1


def load_manual_event():
    try:
        with open("manual.json") as f:
            m = json.load(f)
        ev = m.get("event", {"type": "none", "weeks": 0})
        if ev.get("type") not in ("BOS", "MSB", "none"):
            ev = {"type": "none", "weeks": 0}
        return ev
    except FileNotFoundError:
        return {"type": "none", "weeks": 0}
    except Exception as e:  # noqa: BLE001
        print(f"  [manual] could not read manual.json: {e}", file=sys.stderr)
        return {"type": "none", "weeks": 0}


# ======================================================================
# Assemble
# ======================================================================

def build(raw_overrides=None):
    """Assemble the full view-model. raw_overrides lets --demo inject mock values."""
    now = datetime.now(timezone.utc)

    if raw_overrides is None:
        klines = fetch_klines()
        ind = compute_indicators(klines)
        onchain = fetch_onchain()
    else:
        ind = raw_overrides["ind"]
        onchain = raw_overrides["onchain"]

    event = load_manual_event()

    raw = {
        "week": ind["week"],
        "dayOfWeek": ind["dayOfWeek"],
        "year": cycle_year(now),
        "price": ind["price"],
        "vwapSigma": ind["vwapSigma"],
        "rsi": ind["rsi"],
        "dev200": ind["dev200"],
        "above50W": ind["above50W"],
        "ma3550": ind["ma3550"],
        "p35": ind["p35"], "z35": ind["z35"],
        "p50": ind["p50"], "z50": ind["z50"],
        "sth": onchain.get("sth"),
        "lth": onchain.get("lth"),
        "mvrvZ": onchain.get("mvrvZ"),
        "nupl": onchain.get("nupl"),
        "avgBuyPrice": onchain.get("avgBuyPrice"),
        "event": event,
    }

    # scoring needs these keys present
    score_input = {
        "vwapSigma": raw["vwapSigma"], "rsi": raw["rsi"], "dev200": raw["dev200"],
        "above50W": raw["above50W"], "ma3550": raw["ma3550"], "price": raw["price"],
        "sth": raw["sth"], "lth": raw["lth"], "mvrvZ": raw["mvrvZ"],
        "year": raw["year"], "event": raw["event"],
    }
    missing = [k for k in ("sth", "lth", "mvrvZ") if score_input[k] is None]
    if missing:
        raise RuntimeError(
            f"On-chain values missing {missing} — scoring can't run. "
            f"Run `python fetch.py --probe` to check the BGeometrics endpoints."
        )

    scored = scoring.aggregate(score_input)

    return {
        "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "meta": {
            "anchor": ind.get("_anchor"),
            "maType": MA_TYPE,
            "zLookback": Z_LOOKBACK,
            "provisional": raw["dayOfWeek"] < 7,
        },
        "raw": raw,
        "scored": scored,
    }


# ======================================================================
# Demo (offline) — mid-cycle-ish mock so the dashboard renders without network
# ======================================================================

DEMO = {
    "ind": {
        "price": 118200, "week": "Jun 22 2026", "dayOfWeek": 4,
        "vwapSigma": 0.8, "rsi": 58.0, "dev200": 85.0, "above50W": True,
        "ma3550": "above", "p35": 12.9, "z35": 1.04, "p50": 22.6, "z50": 0.98,
        "_anchor": "2022-11-20",
    },
    "onchain": {"mvrvZ": 2.2, "sth": 95000, "lth": 58000, "nupl": 0.55, "avgBuyPrice": 72000},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="diagnose endpoints")
    ap.add_argument("--demo", action="store_true", help="write data.json from mock values")
    args = ap.parse_args()

    if args.probe:
        print("Probing price source (Binance)…")
        try:
            kl = fetch_klines()
            print(f"  [price] OK — {len(kl)} weekly candles, "
                  f"latest close ${kl[-1]['close']:,.0f} @ {kl[-1]['open_time'].date()}")
        except Exception as e:  # noqa: BLE001
            print(f"  [price] FAILED — {e}")
        print("Probing on-chain source (BGeometrics)…")
        fetch_onchain(probe=True)
        print("\nProbe done. Fix any non-200 slugs in ENDPOINTS, then run without --probe.")
        return

    payload = build(raw_overrides=DEMO if args.demo else None)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    s = payload["scored"]
    tag = " (DEMO)" if args.demo else ""
    print(f"Wrote {OUT_PATH}{tag} — {s['call']} (composite {s['composite']:+.1f}), "
          f"phase: {s['phase']}, regime: {'BULL' if s['bull'] else 'BEAR'}")


if __name__ == "__main__":
    main()
