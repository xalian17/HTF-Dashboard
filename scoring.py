"""
scoring.py — BTC Weekly Cycle Compass scoring engine.

This is a faithful port of the JavaScript aggregate() that was iterated on and
approved in the prototype. It is the SINGLE SOURCE OF TRUTH for scoring: fetch.py
computes raw indicator values, passes them here, and the dashboard only renders the
result. Keeping scoring in one place (Python) avoids JS/Python drift.

Every threshold here is intentionally easy to find and tweak. Nothing about moving to
live data locks these numbers.

Input dict `d` expects:
    vwapSigma : float   price deviation from cycle-anchored W1 VWAP, in sigma
    rsi       : float   W1 RSI (Wilder, 14)
    dev200    : float   % distance of price above/below the 200W MA
    above50W  : bool    price above the 50W MA (regime)
    ma3550    : str     one of "reclaim" | "watching" | "below" | "above"
    price     : float   current price
    sth       : float   short-term-holder realized price (cost basis)
    lth       : float   long-term-holder realized price (cost basis)
    mvrvZ     : float   MVRV Z-Score
    year      : int     year in cycle (1 = bottom year ... 4 = top year)
    event     : dict    {"type": "BOS"|"MSB"|"none", "weeks": int}
"""

from typing import Dict, Any


# ----------------------------------------------------------------------
# Per-indicator scorers  (-2 .. +2 scale; + = accumulate, - = distribute)
# ----------------------------------------------------------------------

def score_vwap(s: float) -> Dict[str, Any]:
    if s >= 2.0:
        return {"score": -2, "zone": "OVERBOUGHT"}
    if s >= 0.0:
        return {"score": +1, "zone": "BUILDING · BULL"}
    if s > -1.0:
        return {"score": 0, "zone": "BUILDING · BEAR"}
    if s > -1.5:
        return {"score": +1, "zone": "APPROACHING O/S"}
    return {"score": +2, "zone": "OVERSOLD"}


def score_rsi(x: float) -> Dict[str, Any]:
    if x >= 70:
        return {"score": -2, "zone": "POSSIBLE TOP"}
    if x >= 50:
        return {"score": 0, "zone": "NEUTRAL"}
    if x >= 30:
        return {"score": +1, "zone": "ACCUMULATE"}
    return {"score": +2, "zone": "AGGR. ACCUM."}


def score_200w(dev: float) -> Dict[str, Any]:
    if dev <= 0:
        return {"score": +2}
    if dev <= 50:
        return {"score": +1}
    if dev <= 150:
        return {"score": 0}
    if dev <= 300:
        return {"score": -1}
    return {"score": -2}


def event_contribution(ev: Dict[str, Any]) -> Dict[str, Any]:
    """BOS: +2 base, decays over 8 wks. MSB: -2 base, decays over 2 wks."""
    if not ev or ev.get("type") == "none":
        return {"contrib": 0.0, "potency": 0.0, "type": "none", "weeks": 0}
    t = ev.get("type")
    w = ev.get("weeks", 0)
    if t == "BOS":
        p = max(0.0, 1 - w / 8)
        return {"contrib": +2 * p, "potency": p, "type": "BOS", "weeks": w}
    if t == "MSB":
        p = max(0.0, 1 - w / 2)
        return {"contrib": -2 * p, "potency": p, "type": "MSB", "weeks": w}
    return {"contrib": 0.0, "potency": 0.0, "type": "none", "weeks": 0}


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


def score_cost_basis(price: float, sth: float, lth: float) -> Dict[str, Any]:
    sth_prem = (price - sth) / sth
    lth_prem = (price - lth) / lth
    if lth_prem < 0:
        score, zone = +2, "BELOW LTH · DEEP VALUE"
    elif price < sth * 0.95:
        score, zone = +1, "BELOW STH · STACKING"
    elif price <= sth * 1.10:
        score, zone = +1, "AT STH · SUPPORT TEST"
    elif price <= sth * 1.60:
        score, zone = 0, "ABOVE STH · HEALTHY"
    else:
        score, zone = -1, "STH PREMIUM · FROTHY"
    return {"score": score, "zone": zone, "sthPrem": sth_prem, "lthPrem": lth_prem}


# ----------------------------------------------------------------------
# Aggregation
#   technical core (VWAP + RSI + 200W + reclaim + event)  × 50W regime gate
#   + on-chain core (cost-basis + MVRV-Z)  added AFTER the gate
# ----------------------------------------------------------------------

def aggregate(d: Dict[str, Any]) -> Dict[str, Any]:
    v = score_vwap(d["vwapSigma"])
    r = score_rsi(d["rsi"])
    m2 = score_200w(d["dev200"])
    cb = score_cost_basis(d["price"], d["sth"], d["lth"])
    mz = score_mvrv_z(d["mvrvZ"])
    ev = event_contribution(d.get("event"))

    bull = bool(d["above50W"])
    reclaim = (d["ma3550"] == "reclaim")
    watching = (d["ma3550"] == "watching")
    re_bonus = +1 if reclaim else 0

    core_price = v["score"] + r["score"] + m2["score"]      # -6 .. +6
    core_chain = cb["score"] + mz["score"]                  # -3 .. +4
    tech_pre = core_price + re_bonus + ev["contrib"]        # pre-gate

    if bull:
        mult = 1.2 if tech_pre >= 0 else 1.0
        mult_label = "Bull · tech buys ×1.2"
    else:
        mult = 0.7 if tech_pre >= 0 else 1.2
        mult_label = "Bear · tech buys ×0.7 (knife-catch discount)"

    tech_gated = tech_pre * mult
    composite = tech_gated + core_chain                     # on-chain un-gated

    if composite >= 8:
        call, kind = "AGGRESSIVE ACCUMULATE", "pos"
    elif composite >= 3.5:
        call, kind = "ACCUMULATE", "pos"
    elif composite > -3.5:
        call, kind = "NEUTRAL · BALANCE", "zero"
    elif composite > -8:
        call, kind = "DISTRIBUTE", "neg"
    else:
        call, kind = "AGGRESSIVE SELL", "neg"

    year = d["year"]
    if year == 1 and (d["vwapSigma"] <= -1.0 or cb["lthPrem"] < 0) and d["mvrvZ"] <= 1:
        cycle = "Year 1 · Cycle Bottom Zone"
    elif year >= 4 and (d["vwapSigma"] >= 1.5 or d["mvrvZ"] >= 5) and d["dev200"] >= 150:
        cycle = "Year %d · Cycle Top Zone" % year
    elif d["mvrvZ"] >= 5:
        cycle = "Euphoria · Distribution Risk"
    elif d["mvrvZ"] <= 0:
        cycle = "Deep Value · Generational"
    elif d["vwapSigma"] >= 0:
        cycle = "Building Value · Expansion"
    else:
        cycle = "Building Value · Pullback"

    return {
        "v": v, "r": r, "m2": m2, "cb": cb, "mz": mz, "ev": ev,
        "reBonus": re_bonus, "corePrice": core_price, "coreChain": core_chain,
        "techPre": tech_pre, "mult": mult, "multLabel": mult_label,
        "techGated": tech_gated, "composite": composite,
        "call": call, "callKind": kind, "cycle": cycle,
        "bull": bull, "reclaim": reclaim, "watching": watching,
    }


if __name__ == "__main__":
    # Quick self-check against the approved JS prototype values.
    import json
    SCEN = {
        "bottom":  dict(year=1, vwapSigma=-1.7, rsi=28, dev200=-5,  above50W=False, ma3550="below",
                        price=61400,  sth=74000,  lth=63000, mvrvZ=0.1, event={"type": "BOS", "weeks": 2}),
        "mid":     dict(year=2, vwapSigma=0.8,  rsi=58, dev200=85,  above50W=True,  ma3550="above",
                        price=118200, sth=95000,  lth=58000, mvrvZ=2.2, event={"type": "none"}),
        "pullback":dict(year=2, vwapSigma=-0.6, rsi=44, dev200=30,  above50W=True,  ma3550="watching",
                        price=87000,  sth=84000,  lth=54000, mvrvZ=1.4, event={"type": "none"}),
        "reclaim": dict(year=2, vwapSigma=-0.3, rsi=46, dev200=40,  above50W=True,  ma3550="reclaim",
                        price=92800,  sth=91000,  lth=56000, mvrvZ=1.6, event={"type": "BOS", "weeks": 4}),
        "top":     dict(year=4, vwapSigma=2.3,  rsi=74, dev200=320, above50W=True,  ma3550="above",
                        price=214500, sth=130000, lth=72000, mvrvZ=5.8, event={"type": "MSB", "weeks": 1}),
    }
    for k, d in SCEN.items():
        a = aggregate(d)
        print(f"{k:9s} comp={a['composite']:+5.1f}  {a['call']:<22s}  "
              f"reBonus={a['reBonus']}  {a['cb']['zone']}")
