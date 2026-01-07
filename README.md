# Profitable Liquidity Provision: Mitigating DeFi Impermanent Loss — Data & Code (Anonymous)

This repository contains the raw pool data and Python scripts used to produce the empirical figures/tables for the accompanying paper submission (double-blind / anonymized).

## Contents

### Raw data
- `pool_data/updated/0x*.json`  
  Daily `poolDayDatas` exports (Uniswap v3 pools) collected via The Graph subgraph explorer. Each file corresponds to one pool address.

### Generated / processed outputs (created by scripts)
The scripts write outputs to folders at the repo root, for example:
- `tvl_analysis/tvl_monthly_end.dat` (monthly end-of-month TVL in USD)
- `trading_volume/trading_volume_monthly.dat` (monthly summed trading volume in USD)
- `monthly_il/monthly_il.dat` (monthly IL series per pool)
- `half_life/halflife_stats.csv` and `half_life/halflife_stats.dat` (mean-reversion half-life + AR(1) phi)
- `backtesting/` (exceedance rates, normalized scores, plots)
- `monthly_R/` (monthly price-ratio summaries)
- `metric_analysis/` (IL metric summaries + TikZ snippet)

## Setup

### Python environment
Tested with Python 3.10+.

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
