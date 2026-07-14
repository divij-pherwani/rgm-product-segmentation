"""Step 1 — clean the raw sales file.  (PySpark)

Three rules. All three come from looking at the data, and all three are explained in
sections 1.1–1.4 of the analysis notebook:

  1. DROP THE PLACEHOLDER ROWS. About 19% of the raw file (16,395 rows) is placeholder rows
     carrying ~1,000 units for a couple of pounds, which works out at roughly 2p a pack.
     Nothing sells at 2p. Price is calculated as revenue / units, so leaving them in makes
     every price number wrong. They are only 0.055% of revenue.

  2. ADD THE BARCODES TOGETHER. The file is really one row per barcode, not per product, so
     different pack sizes of the same product share a week (13,734 rows, 16.2%, are second
     barcodes). Dropping duplicates would delete real sales — every duplicate key has units
     on both rows. So SALES columns are SUMMED. But shelf presence (ACV) takes the MAX, not
     the sum: if the 30-pack is in 50% of shops and the 60-pack in 40%, the product isn't in
     90% of shops — it's largely the same shops.

  3. SET ASIDE THE NEW PRODUCTS. Under a quarter (13 weeks) of history, you can't see a trend
     or a season. These aren't deleted — they're a "watch this space" list (13 products).

WHY SPARK. This is the only step whose cost scales with the number of *rows* rather than the
number of products, so it is the first thing that hurts as the data grows. It is a straight
pass over the rows — filter, then group-and-aggregate — which is exactly what Spark is for.
On the recruitment dataset (84,550 rows) the result is 63,834 rows / 353 products / 157 weeks.
"""
from pyspark.sql import DataFrame, SparkSession, functions as F

# Sales columns are ADDED when two barcodes share a product-retailer-week.
SALES_COLS = ["total_units", "total_volume", "total_sales", "any_promo_units", "any_promo_amt"]

# Availability (ACV) columns are MAX'd, not summed — see rule 2. any_promo_acv_pct is the one
# the promo features are built on, so it has to survive the aggregation.
ACV_COLS = ["acv_pct", "any_promo_acv_pct", "acv_tpr_only", "feature_and_display_acv_pct",
            "feature_without_display_acv_pct", "display_without_feature_acv_pct"]

# Catalogue fields that are constant within a product — carried through with first().
META_COLS = ["subcategory_nm", "brand_nm", "uom", "product_unit_size", "product_pack_count"]


def _strip_bom(df: DataFrame) -> DataFrame:
    """The raw CSV is UTF-8-with-BOM, so the first header can arrive as '﻿retailer_nm'."""
    return df.toDF(*[c.replace("﻿", "").strip() for c in df.columns])


def load_raw(spark: SparkSession, path: str) -> DataFrame:
    """Read the raw weekly sales CSV. Kept here so every entry point reads it the same way."""
    df = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .option("encoding", "UTF-8")
          .csv(path))
    return _strip_bom(df)


def run(raw: DataFrame, settings) -> tuple[DataFrame, list]:
    """Returns (clean product-retailer-week sales, list of rgm_ppg set aside as too new)."""
    raw = _strip_bom(raw)

    # Always parse the date. The raw column is m/d/Y as text; left as a string, every sort is
    # lexicographic ("10/5/2025" < "6/25/2023") and every time series comes out scrambled while
    # still looking plausible. Parse it once, up front.
    parsed = raw.withColumn("period_id", F.to_date("period_id", "M/d/yyyy"))
    if parsed.where(F.col("period_id").isNull()).limit(1).count() > 0:
        raise ValueError("some dates in period_id could not be parsed as m/d/Y")

    for c in SALES_COLS + ACV_COLS + ["product_unit_size", "product_pack_count"]:
        parsed = parsed.withColumn(c, F.col(c).cast("double"))

    price_per_pack = F.col("total_sales") / F.when(F.col("total_units") == 0, None)\
                                              .otherwise(F.col("total_units"))
    parsed = parsed.withColumn("price_per_pack", price_per_pack)

    # Rule 1 — drop the placeholder rows.
    floor = float(settings.cleaning.minimum_price_per_pack)
    kept = parsed.where((F.col("price_per_pack") >= floor) & (F.col("total_units") > 0))

    # Rule 2 — one row per product-retailer-week: SUM the sales, MAX the availability.
    meta = kept.groupBy("rgm_ppg").agg(*[F.first(c, ignorenulls=True).alias(c) for c in META_COLS])
    agg = (kept.groupBy("rgm_ppg", "retailer_nm", "period_id")
               .agg(*[F.sum(c).alias(c) for c in SALES_COLS],
                    *[F.max(c).alias(c) for c in ACV_COLS]))
    sales = (agg.join(meta, on="rgm_ppg", how="left")
                .withColumn("price_per_pack", F.col("total_sales") / F.col("total_units")))

    # Rule 3 — set aside products with too little history to judge.
    min_weeks = int(settings.cleaning.minimum_weeks_of_history)
    weeks = sales.groupBy("rgm_ppg").agg(F.countDistinct("period_id").alias("n_weeks"))
    too_new = [r.rgm_ppg for r in weeks.where(F.col("n_weeks") < min_weeks)
                                       .select("rgm_ppg").collect()]

    sales = sales.join(weeks.where(F.col("n_weeks") >= min_weeks).select("rgm_ppg"),
                       on="rgm_ppg", how="inner")
    return sales, sorted(too_new)
