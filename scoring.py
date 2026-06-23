"""
scoring.py — BTC Weekly Cycle Compass scoring engine (revised three-axis model).

PURPOSE: a weekly HTF cycle/bias read — is BTC cheap, neutral, or extended on the
cycle. NOT an entry trigger. The composite sets exposure posture, never timing.

THREE AXES
    value_core  = VWAP_score + 200W_score                 (ungated — valuation)
    trend_core  = RSI_score + structure_event + reclaim
    trend_gated = trend_core × 50W_regime_multiplier
    chain_core  = MVRV_score + LTH_score + 0.5×STH_score   (ungated — on-chain cycle)
    composite   = value_core + trend_gated + chain_core

Key design rules:
  - Value (VWAP, 200W) and on-chain are NEVER gated by the 50W. On-chain cheapness is
    the cycle-bottom signal precisely when price is below trend averages.
  - Only RSI + structure + reclaim are gated (they are momentum/trend, where the 50W
    legitimately changes reliability).
  - STH contributes HALF weight to the composite but is shown at FULL weight on the
    dashboard (a real heat warning even if it only moves the composite a little).
  - Cycle-year is CONTEXT ONLY. It never affects composite, verdict, or the cycle label.

Every threshold is a reasoning-driven default, not fit to history. Easy to tune.
"""

from typing import Dict, Any


# ----------------------------------------------------------------------
# VALUE axis
# ----------------------------------------------------------------------

def score_vwap(s: float) -> Dict[str, Any]:
    """Annual VWAP deviation in sigma. Below = value; above = trend, only a sell when extended."""
    if s >= 3.5:
        return {"score": -2, "zone": "BLOW-OFF EXTENSION"}
    if s >= 2.5:
        return {"score": -1, "zone": "EXTENDED"}
    if s >= 1.0:
        return {"score": 0, "zone": "CONSTRUCTIVE TREND"}
    if s >= -1.0:
        return {"score": 0, "zone": "AT VALUE"}
    if s >= -2.0:
        return {"score": +1, "zone": "VALUE / PULLBACK"}
    return {"score": +2, "zone": "DEEP VALUE"}


def score_200w(dev: float) -> Dict[str, Any]:
    """% distance above the 200W MA. Compressed bands for the modern era."""
    if dev <= 0:
        return {"score": +2, "zone": "AT / BELOW FLOOR"}
    if dev <= 30:
        return {"score": +1, "zone": "NEAR FLOOR"}
    if dev <= 100:
        return {"score": 0, "zone": "MID-CYCLE"}
    if dev <= 200:
        return {"score": -1, "zone": "EXTENDED"}
    return {"score": -2, "zone": "CYCLE-TOP STRETCH"}


# ----------------------------------------------------------------------
# TREND / STRUCTURE axis (gated)
# ----------------------------------------------------------------------

def score_rsi(x: float) -> Dict[str, Any]:
    if x >= 70:
        return {"score": -2, "zone": "DISTRIBUTION RISK"}
    if x >= 60:
        return {"score": -1, "zone": "HEATED"}
    if x >= 45:
        return {"score": 0, "zone": "NEUTRAL"}
    if x >= 30:
        return {"score": +1, "zone": "ACCUMULATE"}
    return {"score": +2, "zone": "AGGR. ACCUM."}


def event_contribution(ev: Dict[str, Any]) -> Dict[str, Any]:
    """BOS: +2, decays over 8 wks. MSB: -2, decays over 2 wks."""
    if not ev or ev.get("type") == "none":
        return {"contrib": 0.0, "potency": 0.0, "type": "none", "weeks": 0}
    t, w = ev.get("type"), ev.get("weeks", 0)
    if t == "BOS":
        p = max(0.0, 1 - w / 8)
        return {"contrib": +2 * p, "potency": p, "type": "BOS", "weeks": w}
    if t == "MSB":
        p = max(0.0, 1 - w / 2)
        return {"contrib": -2 * p, "potency": p, "type": "MSB", "weeks": w}
    return {"contrib": 0.0, "potency": 0.0, "type": "none", "weeks": 0}


# ----------------------------------------------------------------------
# ON-CHAIN axis (ungated). STH and LTH are SEPARATE ladders.
# ----------------------------------------------------------------------

def score_mvrv_z(z: float) -> Dict[str, Any]:
    if z <= 0:
        return {"score": +2, "zone": "DEEP VALUE"}
    if z <= 1:
        return {"score": +1, "zone": "CHEAP"}
    if z <= 3:
        return {"score": 0, "zone": "FAIR VALUE"}
    if z <= 5:
        return {"score": -1, "zone": "EXPENSIVE"}
    return {"score": -2, "zone": "EUPHORIA"}


def score_lth(price: float, lth: float) -> Dict[str, Any]:
    """LTH realized price = slow, long-cycle cost basis. Price can run far above it."""
    prem = (price / lth - 1) * 100
    if prem < 0:
        score, zone = +2, "BELOW LTH · DEEP VALUE"
    elif prem <= 100:
        score, zone = +1, "CONSTRUCTIVE VALUE"
    elif prem <= 200:
        score, zone = 0, "BULL-CYCLE EXPANSION"
    elif prem <= 300:
        score, zone = -1, "CYCLE EXTENSION"
    else:
        score, zone = -2, "MAJOR EXTENSION"
    return {"score": score, "zone": zone, "prem": prem}


def score_sth(price: float, sth: float) -> Dict[str, Any]:
    """STH realized price = fast, recent-holder cost basis. Tight bands — heat/support."""
    prem = (price / sth - 1) * 100
    if prem <= 10:            # below STH or hugging it (±10% support band)
        score, zone = +1, "AT / BELOW STH · SUPPORT"
    elif prem <= 20:
        score, zone = 0, "HEALTHY"
    elif prem <= 40:
        score, zone = -1, "HEATED"
    else:
        score, zone = -2, "OVERHEATED PROFIT"
    return {"score": score, "zone": zone, "prem": prem}


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def aggregate(d: Dict[str, Any]) -> Dict[str, Any]:
    vwap = score_vwap(d["vwapSigma"])
    w200 = score_200w(d["dev200"])
    rsi = score_rsi(d["rsi"])
    mvrv = score_mvrv_z(d["mvrvZ"])
    lth = score_lth(d["price"], d["lth"])
    sth = score_sth(d["price"], d["sth"])
    ev = event_contribution(d.get("event"))

    bull = bool(d["above50W"])
    reclaim = (d["ma3550"] == "reclaim")
    watching = (d["ma3550"] == "watching")
    re_bonus = +1 if reclaim else 0

    # --- value axis (ungated) ---
    value_core = vwap["score"] + w200["score"]                       # -4 .. +4

    # --- trend axis (gated by 50W) ---
    trend_core = rsi["score"] + ev["contrib"] + re_bonus
    if bull:
        mult = 1.15 if trend_core >= 0 else 1.0
        mult_label = "Bull · trend buys ×1.15"
    else:
        mult = 0.85 if trend_core >= 0 else 1.15
        mult_label = "Bear · trend buys ×0.85"
    trend_gated = trend_core * mult

    # --- on-chain axis (ungated). STH at half weight in the composite. ---
    sth_weighted = 0.5 * sth["score"]
    chain_core = mvrv["score"] + lth["score"] + sth_weighted         # ~ -5 .. +5

    composite = value_core + trend_gated + chain_core

    # --- verdict (asymmetric: gives up green easier than it earns it) ---
    if composite >= 8:
        call, kind = "AGGRESSIVE ACCUMULATE", "pos"
    elif composite >= 3.5:
        call, kind = "ACCUMULATE", "pos"
    elif composite > -3:
        call, kind = "NEUTRAL", "zero"
    elif composite > -7:
        call, kind = "DISTRIBUTE", "neg"
    else:
        call, kind = "AGGRESSIVE SELL", "neg"

    # --- cycle phase from EVIDENCE ONLY (no cycle-year gate) ---
    if d["mvrvZ"] < 0 or lth["prem"] < 0:
        cycle = "Deep Value · Generational"
    elif d["mvrvZ"] >= 5 or d["vwapSigma"] >= 3.5 or lth["prem"] > 300:
        cycle = "Euphoria · Distribution Risk"
    elif value_core >= 3 and chain_core >= 2:
        cycle = "Cycle Bottom Zone"
    elif value_core <= -2 and chain_core <= -2:
        cycle = "Cycle Top Zone"
    elif composite > 0 and bull:
        cycle = "Building Value · Expansion"
    elif composite > 0:
        cycle = "Building Value · Pullback"
    else:
        cycle = "Mid-Cycle · Neutral"

    # --- cycle-year: CONTEXT NOTE ONLY. Does not touch score/verdict/label. ---
    yr = d.get("year")
    if yr is None:
        cycle_note = ""
    elif yr >= 4:
        cycle_note = f"Cycle age: year {yr} from last low · atypical / lower confidence"
    else:
        cycle_note = f"Cycle age: year {yr} from last low · typical window"

    return {
        # per-indicator (each has .score and .zone; lth/sth also .prem)
        "vwap": vwap, "w200": w200, "rsi": rsi, "mvrv": mvrv,
        "lth": lth, "sth": sth, "ev": ev, "reBonus": re_bonus,
        # axes
        "valueCore": value_core,
        "trendCore": trend_core, "trendGated": trend_gated,
        "chainCore": chain_core, "sthWeighted": sth_weighted,
        "mult": mult, "multLabel": mult_label,
        "composite": composite,
        # verdict + labels
        "call": call, "callKind": kind, "cycle": cycle, "cycleNote": cycle_note,
        "bull": bull, "reclaim": reclaim, "watching": watching,
    }


if __name__ == "__main__":
    # Sanity scenarios (NOT fit to history) — confirm sane, monotonic behavior.
    SCEN = {
        "deep_bottom": dict(year=1, vwapSigma=-2.1, rsi=26, dev200=-8, above50W=False, ma3550="below",
                            price=58000, sth=74000, lth=55500, mvrvZ=-0.2, event={"type": "BOS", "weeks": 2}),
        "value_pullback": dict(year=2, vwapSigma=-1.2, rsi=42, dev200=20, above50W=True, ma3550="reclaim",
                               price=64000, sth=68000, lth=40000, mvrvZ=0.8, event={"type": "none"}),
        "mid_cycle": dict(year=2, vwapSigma=0.6, rsi=55, dev200=70, above50W=True, ma3550="above",
                          price=95000, sth=82000, lth=42000, mvrvZ=2.1, event={"type": "none"}),
        "hot_midcycle": dict(year=3, vwapSigma=1.4, rsi=72, dev200=120, above50W=True, ma3550="above",
                             price=140000, sth=96000, lth=46000, mvrvZ=3.2, event={"type": "none"}),
        "euphoria_top": dict(year=4, vwapSigma=3.2, rsi=78, dev200=210, above50W=True, ma3550="above",
                             price=210000, sth=120000, lth=48000, mvrvZ=5.4, event={"type": "MSB", "weeks": 1}),
    }
    for k, d in SCEN.items():
        a = aggregate(d)
        print(f"{k:16s} comp={a['composite']:+6.2f}  {a['call']:<22s}  "
              f"value={a['valueCore']:+d} trendG={a['trendGated']:+.2f} chain={a['chainCore']:+.1f}  "
              f"| LTH {a['lth']['score']:+d} STH {a['sth']['score']:+d}(½→{a['sthWeighted']:+.1f})  | {a['cycle']}")
