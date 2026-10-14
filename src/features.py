"""Spark feature engineering for stock + prediction-market data.

Reads the raw Parquet files from data_ingest.py, joins prices with
prediction-market signals, and builds the features the models use.

Key point: every feature for day t uses only data up to and including t.
The target (next-day return) is built with a lead, so there is no
lookahead leakage.

Output: data/features/features.parquet
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src import config


def get_spark() -> SparkSession:
    """Build a local Spark session with sensible defaults."""
    return (
        SparkSession.builder.appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", str(config.SPARK_SHUFFLE_PARTITIONS))
        .getOrCreate()
    )


def add_technical_features(df: DataFrame) -> DataFrame:
    """Add moving averages, RSI, volatility, and lagged returns."""
    # Window per ticker, ordered by date. We use rowsBetween so the
    # calculation is purely based on past rows.
    w = Window.partitionBy("ticker").orderBy("date")

    # Daily return, computed from close prices.
    df = df.withColumn("prev_close", F.lag("close", 1).over(w))
    df = df.withColumn(
        "daily_return",
        F.when(F.col("prev_close").isNotNull(), (F.col("close") - F.col("prev_close")) / F.col("prev_close")),
    )

    # Moving averages and volume trends.
    for window in config.MA_WINDOWS:
        win = w.rowsBetween(-window + 1, 0)
        df = df.withColumn(f"ma_{window}", F.avg("close").over(win))
        df = df.withColumn(f"ma_ratio_{window}", F.col("close") / F.col(f"ma_{window}"))
        df = df.withColumn(f"vol_ma_{window}", F.avg("volume").over(win))

    # Volatility: rolling stddev of daily returns.
    vol_win = w.rowsBetween(-config.VOLATILITY_WINDOW + 1, 0)
    df = df.withColumn("volatility", F.stddev("daily_return").over(vol_win))

    # RSI (14-day by default). We keep the classic Wilder-style formula
    # simple: average gain / average loss over the window.
    df = df.withColumn("gain", F.when(F.col("daily_return") > 0, F.col("daily_return")).otherwise(0))
    df = df.withColumn("loss", F.when(F.col("daily_return") < 0, -F.col("daily_return")).otherwise(0))
    rsi_win = w.rowsBetween(-config.RSI_WINDOW + 1, 0)
    df = df.withColumn("avg_gain", F.avg("gain").over(rsi_win))
    df = df.withColumn("avg_loss", F.avg("loss").over(rsi_win))
    df = df.withColumn(
        "rsi",
        F.when(
            F.col("avg_loss") == 0,
            F.lit(100.0),
        ).otherwise(100 - (100 / (1 + F.col("avg_gain") / F.col("avg_loss")))),
    )

    # Lagged returns so the model can see recent momentum.
    for lag in config.LAG_DAYS:
        df = df.withColumn(f"return_lag_{lag}", F.lag("daily_return", lag).over(w))

    # Prediction-market features: how far the market prob is from 50/50,
    # and whether the market is confident (tight spread, high volume).
    df = df.withColumn("market_edge", F.col("market_prob_up") - 0.5)
    df = df.withColumn("market_confidence", (1 - F.col("market_spread")) * F.log1p(F.col("market_volume")))

    return df


def add_targets(df: DataFrame) -> DataFrame:
    """Add next-day return and up/down label (the things we predict)."""
    w = Window.partitionBy("ticker").orderBy("date")
    df = df.withColumn("next_close", F.lead("close", 1).over(w))
    df = df.withColumn(
        "next_return",
        F.when(F.col("next_close").isNotNull(), (F.col("next_close") - F.col("close")) / F.col("close")),
    )
    # Binary label: 1 if the stock goes up tomorrow, else 0.
    df = df.withColumn("label_up", (F.col("next_return") > 0).cast("int"))
    # Drop the last row per ticker, where the target is unknown.
    return df.filter(F.col("next_return").isNotNull())


def build_features(spark: SparkSession) -> DataFrame:
    """Load raw data, join, engineer features, and save to Parquet."""
    prices = spark.read.parquet(str(config.RAW_DIR / "prices.parquet"))
    market = spark.read.parquet(str(config.RAW_DIR / "prediction_market.parquet"))

    # Join on date + ticker. Both files share the same keys.
    df = prices.join(market, on=["date", "ticker"], how="inner")
    df = add_technical_features(df)
    df = add_targets(df)

    # Drop rows where early-window features are still null (first few days).
    df = df.dropna(subset=["volatility", "rsi"] + [f"return_lag_{lag}" for lag in config.LAG_DAYS])
    return df


def main() -> None:
    config.ensure_dirs()
    spark = get_spark()
    df = build_features(spark)
    print(f"Built {df.count()} feature rows.")
    df.write.mode("overwrite").parquet(str(config.FEATURES_DIR / "features.parquet"))
    print(f"Wrote {config.FEATURES_DIR / 'features.parquet'}")
    spark.stop()


if __name__ == "__main__":
    main()
