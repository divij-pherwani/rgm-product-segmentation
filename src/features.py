"""Step 2 — turn the weekly sales into one row per product.  (PySpark)

THE ONE IDEA THAT MATTERS HERE.

How much a product sells depends mostly on how many shops stock it (section 1.4 of the
notebook — the correlation of log-shelf with log-units is 0.56). So if you measure "growth"
as the trend in weekly units, you're largely measuring whether it got listed in more shops,
not whether shoppers want it. A product being rolled out looks like a hit; one being delisted
looks like a failure. Opposite decisions. So growth, seasonality and promo-lift are all
measured on the SALES RATE — units divided by (smoothed) shelf presence. On the real data,
173 of 353 products change the DIRECTION of their growth once you do this — and 97% of those
flips point the same way as the product's shelf trend, which is the check that the shelf
explanation is real rather than convenient.

NO PROMO THRESHOLD ANYWHERE. Promotion is read straight from any_promo_acv_pct (what actually
ran on deal), never from a made-up "counts as promoted above X%" cutoff.

WHY SPARK, AND WHY pandas INSIDE IT.
  * The product-level aggregates (how big, how expensive, how much promoted) are a plain
    distributed group-by — pure Spark.
  * The per-product time-series metrics (growth, seasonality, promo-lift) need the weeks in
    order, one product at a time. Each product is independent of every other, so this is
    embarrassingly parallel: `groupBy("rgm_ppg").applyInPandas(...)` ships one product to one
    core and runs the exact same numpy on it. That is the pattern the notebook's production
    note points at ("the same function, run on 100 machines instead of one").
  * The numeric core lives in the pure, engine-free functions `weekly_view`, `movement` and
    `price_tier_within_group` below, so the maths is identical to the pandas notebook and can
    be unit-tested with no SparkSession (see tests/test_pipeline.py).
"""
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

# Numeric core (engine-free) lives in _core so Spark and the tests share one definition.
from src._core import (FEATURE_NAMES, movement, price_tier_within_group,  # noqa: F401
                       weekly_view)

# --------------------------------------------------------------------------------------------
# Spark plumbing.
# --------------------------------------------------------------------------------------------
_MOVE_SCHEMA = StructType([
    StructField("rgm_ppg", StringType()),
    StructField("growth", DoubleType()),
    StructField("shelf_trend", DoubleType()),
    StructField("seasonality", DoubleType()),
    StructField("promo_lift", DoubleType()),
])


def _movement_udf(smoothing_weeks, min_weeks_trend, min_weeks_promo_lift):
    """Bind the config scalars into a picklable grouped-map function."""
    def fn(pdf: pd.DataFrame) -> pd.DataFrame:
        m = movement(pdf, smoothing_weeks, min_weeks_trend, min_weeks_promo_lift)
        return pd.DataFrame([{"rgm_ppg": pdf.rgm_ppg.iloc[0], **m}])
    return fn


def run(sales: DataFrame, settings) -> DataFrame:
    """One row per product: size, price, availability, dynamics and promo response."""
    f = settings.features

    # --- product-level aggregates: a plain distributed group-by ---
    prod = (sales.groupBy("rgm_ppg", "subcategory_nm", "brand_nm", "uom")
            .agg(F.avg("total_units").alias("avg_units"),
                 F.stddev_samp("total_units").alias("unit_swing"),
                 F.avg("price_per_pack").alias("avg_price"),
                 F.stddev_samp("price_per_pack").alias("price_swing"),
                 F.sum("total_sales").alias("total_revenue"),
                 F.avg("acv_pct").alias("shelf_presence"),
                 F.sum("any_promo_amt").alias("promo_revenue"),
                 F.avg("any_promo_acv_pct").alias("promo_activity"),
                 F.first("product_unit_size", ignorenulls=True).alias("pack_size"),
                 F.first("product_pack_count", ignorenulls=True).alias("pack_count"),
                 F.countDistinct("period_id").alias("n_weeks"))
            .withColumn("bumpiness", F.col("unit_swing") / F.col("avg_units"))
            .withColumn("price_moves", F.col("price_swing") / F.col("avg_price"))
            .withColumn("promo_share", F.col("promo_revenue") / F.col("total_revenue")))

    # --- price_tier: percentile rank per (subcategory, uom); pandas rank inside Spark ---
    tier_schema = StructType(prod.schema.fields + [StructField("price_tier", DoubleType())])
    prod = (prod.groupBy("subcategory_nm", "uom")
            .applyInPandas(lambda p: price_tier_within_group(p), schema=tier_schema))

    # --- per-product dynamics: one product to one core, exact notebook numpy ---
    move = (sales.select("rgm_ppg", "period_id", "total_units", "acv_pct", "any_promo_acv_pct")
            .groupBy("rgm_ppg")
            .applyInPandas(_movement_udf(int(f.shelf_smoothing_weeks),
                                         int(f.minimum_weeks_for_trend),
                                         int(f.minimum_weeks_for_promo_lift)),
                           schema=_MOVE_SCHEMA))

    return prod.join(move, on="rgm_ppg", how="left")
