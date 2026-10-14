"""Shared configuration for the stock + prediction-market pipeline.

All paths are relative to the repo root. Override tickers / dates with
environment variables so you can run a quick experiment without editing code:

    TICKERS="NVDA,TSLA" START_DATE="2022-01-01" python -m src.data_ingest
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root is one level up from this file (src/ -> repo root).
ROOT = Path(__file__).resolve().parent.parent

# Data layout. These directories are created on first run and gitignored.
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = ROOT / "models"

# Market data settings.
TICKERS = os.getenv("TICKERS", "AAPL,MSFT,SPY").split(",")
# Keep tickers clean in case someone writes "AAPL, MSFT" with a space.
TICKERS = [t.strip().upper() for t in TICKERS if t.strip()]

START_DATE = os.getenv("START_DATE", "2023-01-01")
END_DATE = os.getenv("END_DATE", "2025-01-01")

# Spark settings. Local mode is fine for a portfolio-sized dataset.
SPARK_APP_NAME = "stock-prediction-market"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_SHUFFLE_PARTITIONS = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))

# Train / test split. We split by time, not randomly, to avoid leaking
# future information into training.
TRAIN_FRACTION = 0.8

# Feature engineering windows.
MA_WINDOWS = [5, 10, 20]
RSI_WINDOW = 14
VOLATILITY_WINDOW = 20
LAG_DAYS = [1, 2, 3, 5]

# Prediction-market mock settings.
# The mock generator is seeded so runs are reproducible.
MOCK_MARKET_SEED = 42
# How strongly the mock market probability follows recent momentum.
MOCK_MARKET_MOMENTUM_WEIGHT = 0.6

# Model settings.
RANDOM_SEED = 42
RF_NUM_TREES = 100
GBT_MAX_ITER = 50
GBT_MAX_DEPTH = 5


def ensure_dirs() -> None:
    """Create data / model directories if they do not exist yet."""
    for d in (RAW_DIR, FEATURES_DIR, PREDICTIONS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
