"""Train Spark ML models: regression for next-day return, classification for up/down.

We use a time-based split (first 80% of dates for training, last 20% for
testing) because a random split would leak future data into training.

Models:
- GradientBoostedTrees regressor -> predicts next-day return
- RandomForest classifier     -> predicts up (1) / down (0)

Outputs:
    models/regression_model/      - saved GBT model
    models/classification_model/  - saved RF model
    models/metrics.json           - evaluation metrics on the test set
"""
from __future__ import annotations

import json

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import config
from src.features import get_spark

# All numeric columns the models get to see. Keep this list in sync with
# src/features.py so new features are picked up here automatically.
FEATURE_COLS = (
    [f"ma_ratio_{w}" for w in config.MA_WINDOWS]
    + [f"vol_ma_{w}" for w in config.MA_WINDOWS]
    + [f"return_lag_{lag}" for lag in config.LAG_DAYS]
    + [
        "daily_return",
        "volatility",
        "rsi",
        "market_prob_up",
        "market_edge",
        "market_confidence",
        "market_spread",
    ]
)


def time_split(df: DataFrame, fraction: float = 0.8) -> tuple[DataFrame, DataFrame]:
    """Split by date: earliest dates train, latest dates test."""
    # One global cutoff date keeps the split consistent across tickers.
    cutoff = df.select(F.expr(f"percentile_approx(date, {fraction})").alias("cutoff")).collect()[0]["cutoff"]
    train = df.filter(F.col("date") <= cutoff)
    test = df.filter(F.col("date") > cutoff)
    return train, test


def build_regression_pipeline() -> Pipeline:
    """GBT regression pipeline for next-day return."""
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="assembled", handleInvalid="skip")
    scaler = StandardScaler(inputCol="assembled", outputCol="features")
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol="next_return",
        maxIter=config.GBT_MAX_ITER,
        maxDepth=config.GBT_MAX_DEPTH,
        seed=config.RANDOM_SEED,
    )
    return Pipeline(stages=[assembler, scaler, gbt])


def build_classification_pipeline() -> Pipeline:
    """RandomForest pipeline for up/down direction."""
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="assembled", handleInvalid="skip")
    scaler = StandardScaler(inputCol="assembled", outputCol="features")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label_up",
        numTrees=config.RF_NUM_TREES,
        seed=config.RANDOM_SEED,
    )
    return Pipeline(stages=[assembler, scaler, rf])


def evaluate(df: DataFrame, reg_model, clf_model) -> dict:
    """Compute metrics on the held-out test set."""
    reg_pred = reg_model.transform(df)
    clf_pred = clf_model.transform(df)

    reg_eval_rmse = RegressionEvaluator(labelCol="next_return", predictionCol="prediction", metricName="rmse")
    reg_eval_mae = RegressionEvaluator(labelCol="next_return", predictionCol="prediction", metricName="mae")
    clf_eval_acc = BinaryClassificationEvaluator(labelCol="label_up", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    accuracy = clf_pred.filter(F.col("label_up") == F.col("prediction")).count() / max(clf_pred.count(), 1)

    return {
        "regression_rmse": round(reg_eval_rmse.evaluate(reg_pred), 6),
        "regression_mae": round(reg_eval_mae.evaluate(reg_pred), 6),
        "classification_accuracy": round(accuracy, 4),
        "classification_auc": round(clf_eval_acc.evaluate(clf_pred), 4),
        "test_rows": df.count(),
    }


def main() -> None:
    config.ensure_dirs()
    spark = get_spark()
    df = spark.read.parquet(str(config.FEATURES_DIR / "features.parquet"))
    print(f"Loaded {df.count()} feature rows.")

    train, test = time_split(df, config.TRAIN_FRACTION)
    print(f"Train rows: {train.count()}, test rows: {test.count()}")
    train = train.cache()

    reg_model = build_regression_pipeline().fit(train)
    print("Trained GBT regression model.")
    clf_model = build_classification_pipeline().fit(train)
    print("Trained RF classification model.")

    metrics = evaluate(test, reg_model, clf_model)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    reg_model.write().overwrite().save(str(config.MODELS_DIR / "regression_model"))
    clf_model.write().overwrite().save(str(config.MODELS_DIR / "classification_model"))
    with open(config.MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved models and metrics to {config.MODELS_DIR}")
    spark.stop()


if __name__ == "__main__":
    main()
