"""Batch inference: load trained models and predict on the latest features.

Reads the feature Parquet from src/features.py, loads the models saved by
src/train.py, and writes predictions to Parquet plus a small summary CSV.

Outputs:
    data/predictions/predictions.parquet - full prediction rows
    data/predictions/summary.csv         - one row per ticker: latest date,
                                           predicted return, predicted direction
"""
from __future__ import annotations

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from src import config
from src.features import get_spark


def load_models() -> tuple[PipelineModel, PipelineModel]:
    """Load the saved regression and classification models."""
    reg = PipelineModel.load(str(config.MODELS_DIR / "regression_model"))
    clf = PipelineModel.load(str(config.MODELS_DIR / "classification_model"))
    return reg, clf


def run_inference(spark: SparkSession):
    """Score the latest feature row per ticker and save results."""
    features = spark.read.parquet(str(config.FEATURES_DIR / "features.parquet"))
    reg_model, clf_model = load_models()

    # Score everything; the summary below picks the most recent row.
    scored = reg_model.transform(features)
    # Regression prediction lands in "prediction"; rename it so the
    # classifier's "prediction" column does not collide.
    scored = scored.withColumnRenamed("prediction", "predicted_return")
    scored = clf_model.transform(scored)
    scored = scored.withColumnRenamed("prediction", "predicted_up")

    scored.write.mode("overwrite").parquet(str(config.PREDICTIONS_DIR / "predictions.parquet"))

    # Latest row per ticker gives a clean "what does the model say now" view.
    latest = (
        scored.withColumn(
            "rn",
            F.row_number().over(Window.partitionBy("ticker").orderBy(F.col("date").desc())),
        )
        .filter(F.col("rn") == 1)
        .select(
            "ticker",
            "date",
            "close",
            "predicted_return",
            "predicted_up",
            F.col("probability").getItem(1).alias("prob_up"),
        )
    )
    latest.toPandas().to_csv(config.PREDICTIONS_DIR / "summary.csv", index=False)
    return latest


def main() -> None:
    config.ensure_dirs()
    spark = get_spark()
    latest = run_inference(spark)
    print("Latest predictions:")
    latest.show(truncate=False)
    print(f"Wrote predictions to {config.PREDICTIONS_DIR}")
    spark.stop()


if __name__ == "__main__":
    main()
