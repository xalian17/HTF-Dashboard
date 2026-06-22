# BTC Weekly Cycle Compass

Daily-refreshing dashboard scoring BTC against 6 weekly inputs (VWAP σ, RSI, 35/50/200 MAs,
MSB structure, MVRV-Z, STH/LTH cost basis) into a composite cycle read.

## First run (do this once)
```
pip install -r requirements.txt
python fetch.py --probe      # checks both data sources, prints URL+status for each
```
Binance (price) should work as-is. If a BGeometrics on-chain slug returns non-200,
fix it in the ENDPOINTS dict at the top of fetch.py, then:
```
python fetch.py              # writes data.json
```
Open index.html. (Use a local server or just open the file — it falls back to a sample
if data.json isn't served.)

## Deploy (auto-updates 00:00 UTC)
1. Push this folder to a GitHub repo.
2. Settings → Pages → Source: GitHub Actions.
3. The workflow runs daily, commits data.json, redeploys. Trigger manually anytime via
   Actions → Update BTC Compass → Run workflow.

## Marking a structure event (MSB/BOS)
Edit manual.json, set type to BOS/MSB and weeks-ago, commit. See its _help.

## Tuning
All scoring thresholds live in scoring.py. All data/cycle/MA settings live in the CONFIG
block atop fetch.py (cycle anchors, MA type, z-score window). Nothing is locked.
