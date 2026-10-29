# stockgro-timeseries-capstone

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Capstone](https://img.shields.io/badge/IIT%20Guwahati-StockGro-navy)
![Licence](https://img.shields.io/badge/licence-MIT-lightgrey)

Data driven stock analysis, time series forecasting, and portfolio optimization for the Consulting and Analytics Club, IIT Guwahati, with StockGro. The repository name is unchanged.

Author: Parth Pariwandh

## What it does

- NSE stock universe selection
- Time series forecasting with classical and learning models
- Volatility modelling
- Portfolio optimization and trading signal simulation
- Evaluation dashboards

## Data

Source: Yahoo Finance. Range: 2021-01-01 to 2025-12-31. Final 6 months reserved for testing. Universe: HDFCBANK.NS, TCS.NS, SUNPHARMA.NS, HINDUNILVR.NS, MARUTI.NS, RELIANCE.NS.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
streamlit run app/streamlit_dashboard.py
```

## Licence

MIT.
