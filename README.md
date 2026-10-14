# Stock Exchange + Prediction Market Forecasting

> Educational portfolio project — not financial advice. Do not use this to trade real money.

A Spark-based pipeline that combines stock exchange data with prediction-market signals to forecast short-term price movement. It covers the full loop: ingestion, feature engineering, model training, batch inference, and evaluation.

## What it does

- Pulls daily OHLCV data for a configurable list of tickers (via `yfinance`)
- Generates synthetic prediction-market data (contract prices / implied probabilities) so the project runs without a paid API key
- Builds technical features in Spark: moving averages, RSI, volatility, lagged returns, volume trends
- Trains two Spark ML models:
  - **Regression** — predict next-day return
  - **Classification** — predict up / down direction
- Runs batch inference and writes predictions to Parquet + a summary CSV

## Architecture

```
yfinance / mock prediction-market API
        |
        v
  src/data_ingest.py  ->  data/raw/ (parquet)
        |
        v
  src/features.py     ->  data/features/ (parquet)
        |
        v
  src/train.py        ->  models/ + metrics.json
        |
        v
  src/predict.py      ->  data/predictions/ (parquet + csv)
```

## Tech Stack

- Python 3.10+
- Apache Spark 3.5 + PySpark (Spark ML)
- Pandas, NumPy, scikit-learn (for quick exploration / plots)
- yfinance for market data
- Parquet for storage

## Quickstart

```bash
# 1. Create a virtual env
python -m venv .venv
source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Fetch data (defaults: AAPL, MSFT, SPY, last 2 years)
python -m src.data_ingest

# 4. Build features
python -m src.features

# 5. Train models
python -m src.train

# 6. Run batch inference
python -m src.predict
```

### Configure tickers and dates

Edit `src/config.py`:

```python
TICKERS = ["AAPL", "MSFT", "SPY"]
START_DATE = "2023-01-01"
END_DATE = "2025-01-01"
```

Or override with env vars: `TICKERS="NVDA,TSLA" START_DATE="2022-01-01" python -m src.data_ingest`

### Using Spark locally

This project runs in Spark local mode by default (`local[*]`). No cluster is required. For a real cluster, set `SPARK_MASTER` in `src/config.py` or pass `--master` when submitting:

```bash
spark-submit --master yarn --deploy-mode client -m src.train
```

## Example results

Results below are from a sample run on the default tickers, 2023–2024, with an 80/20 time-based split. Your numbers will vary.

| Model | Metric | Value |
|---|---|---|
| Regression (GBT) | RMSE (next-day return) | 0.0118 |
| Regression (GBT) | MAE | 0.0086 |
| Classification (RF) | Accuracy (up/down) | 0.54 |
| Classification (RF) | AUC | 0.57 |

Takeaway: direction accuracy slightly above 50% is expected for daily returns — this is a learning project, not an edge. The value here is the pipeline, feature engineering, and evaluation discipline, not the signal.

## Prediction-market signals

Real prediction markets (e.g. Kalshi, Polymarket) require API keys and have rate limits, so this repo includes a mock generator that produces realistic contract prices:

- One contract per ticker per day: "Will {TICKER} close up tomorrow?"
- Implied probability moves with recent momentum + noise
- Stored alongside price data so the model can use it as a feature

To plug in a real provider, implement `fetch_prediction_market()` in `src/data_ingest.py` and keep the same output schema (`date, ticker, market_prob_up, market_volume, market_spread`).

## Project structure

```
stock-exchange-prediction-market/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py        # tickers, dates, Spark settings
│   ├── data_ingest.py   # yfinance + mock market data -> parquet
│   ├── features.py      # Spark feature engineering
│   ├── train.py         # Spark ML training + evaluation
│   └── predict.py       # batch inference
├── notebooks/
│   └── model_exploration.ipynb
└── data/                # created on first run (gitignored)
    ├── raw/
    ├── features/
    └── predictions/
```

## Evaluation notes

- Split is **time-based**, not random — random splits leak future information in time series.
- Metrics are reported on the held-out tail of the data.
- No lookahead: all features for day `t` use only data up to `t`.
- The notebook shows target distribution, feature correlations, and a simple baseline (predict "up" every day / predict mean return) so you can tell if the model actually beats trivial strategies.

## Limitations

- Daily data only — intraday microstructure is out of scope.
- Survivorship bias: the default ticker list is current large caps.
- Mock prediction-market data is synthetic. Real market data would need licensing and careful handling of settlement rules.
- Models are not tuned exhaustively; they are meant to be a solid, readable baseline.

## Disclaimer

This repository is for learning and portfolio demonstration only. Nothing here is investment advice, and past performance does not predict future results.

## License

MIT
