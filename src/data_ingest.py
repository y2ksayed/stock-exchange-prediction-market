"""Fetch stock data and prediction-market signals, save as Parquet.

Stock data comes from yfinance (daily OHLCV). Prediction-market data is
mocked here on purpose: real providers need API keys and have rate limits,
and a deterministic mock keeps this project runnable for anyone.

Output:
    data/raw/prices.parquet        - date, ticker, open, high, low, close, volume
    data/raw/prediction_market.parquet - date, ticker, market_prob_up, market_volume, market_spread

To plug in a real prediction-market provider, replace
fetch_prediction_market() and keep the same output schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for each ticker via yfinance."""
    # Import here so unit tests that only use the mock generator
    # do not need yfinance installed.
    import yfinance as yf

    frames = []
    for ticker in tickers:
        # yfinance returns an empty frame for bad tickers / no data,
        # so we skip those instead of crashing the whole run.
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            print(f"No data for {ticker}, skipping.")
            continue
        # yfinance uses a MultiIndex when fetching multiple tickers;
        # flatten it for the single-ticker case.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["ticker"] = ticker
        # Normalize column names to lowercase for consistency downstream.
        df.columns = [c.lower() for c in df.columns]
        frames.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])

    if not frames:
        raise ValueError("No price data downloaded for any ticker.")
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_prediction_market(prices: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate realistic mock prediction-market data.

    For each ticker/day we simulate one contract: "Will {ticker} close up
    tomorrow?" The implied probability follows recent momentum plus noise,
    which is roughly how these markets behave in practice.

    Columns: date, ticker, market_prob_up, market_volume, market_spread.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        # Daily return, shifted so the market on day t does not see
        # the return it is trying to predict.
        returns = group["close"].pct_change().fillna(0)
        momentum = returns.rolling(5, min_periods=1).mean().fillna(0)

        for i, row in group.iterrows():
            # Base probability around 0.5, nudged by momentum.
            # Clip to keep it in a plausible [0.15, 0.85] range.
            prob = 0.5 + config.MOCK_MARKET_MOMENTUM_WEIGHT * float(momentum.iloc[i]) * 5
            prob = float(np.clip(prob + rng.normal(0, 0.05), 0.15, 0.85))
            # Volume is higher on volatile days, which matches real markets.
            volume = int(1000 + abs(float(returns.iloc[i])) * 200_000 + rng.integers(0, 2000))
            spread = float(np.clip(0.01 + abs(rng.normal(0, 0.01)), 0.005, 0.05))
            rows.append(
                {
                    "date": row["date"],
                    "ticker": ticker,
                    "market_prob_up": prob,
                    "market_volume": volume,
                    "market_spread": spread,
                }
            )

    market = pd.DataFrame(rows)
    return market.sort_values(["ticker", "date"]).reset_index(drop=True)


def main() -> None:
    config.ensure_dirs()
    print(f"Fetching prices for {config.TICKERS} from {config.START_DATE} to {config.END_DATE}")
    prices = fetch_prices(config.TICKERS, config.START_DATE, config.END_DATE)
    print(f"Got {len(prices)} price rows.")

    market = fetch_prediction_market(prices, seed=config.MOCK_MARKET_SEED)
    print(f"Generated {len(market)} prediction-market rows.")

    prices_path = config.RAW_DIR / "prices.parquet"
    market_path = config.RAW_DIR / "prediction_market.parquet"
    prices.to_parquet(prices_path, index=False)
    market.to_parquet(market_path, index=False)
    print(f"Wrote {prices_path}")
    print(f"Wrote {market_path}")


if __name__ == "__main__":
    main()
