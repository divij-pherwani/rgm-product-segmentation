"""Step 5 (optional) — which products compete, and which sell together.  (PySpark + numpy)

Two products "compete" if B getting more expensive pushes A's sales UP (a substitute); they
"sell together" if B getting more expensive pulls A's sales DOWN (a complement). We read that
off the sign of B's price in a regression of A's weekly units, controlling for A's own price,
its promotions, its shelf presence, the time trend, and total category demand that week — so a
link is a real cross-price effect, not two products both just riding the January peak.

WHY IT'S SHAPED THE WAY IT IS (the scalability trap).
Comparing every pair is O(n²): double the products and you quadruple the work. Two guards keep
it honest — only compare WITHIN a subcategory (products only substitute for near neighbours),
and only the top-N sellers by volume (the tail is too thin to regress). The weekly aggregation
is distributed in Spark; the pairwise fits then run on the collected top-N series (numpy on the
driver), because a pairwise regression is inherently all-against-all and, bounded to ~40
products, is 780 small fits — tiny. On the real data: 780 pairs -> 45 strict links
(28 compete, 17 sell together) after a Bonferroni cut and an effect-size floor.
"""
import pandas as pd
from pyspark.sql import DataFrame, functions as F

from src._core import links_from_weekly


def _weekly_top_subcategory(sales: DataFrame) -> pd.DataFrame:
    """Distributed: pick the biggest subcategory, build its weekly product table, collect it."""
    top_sub = (sales.groupBy("subcategory_nm").count()
               .orderBy(F.desc("count")).first().subcategory_nm)
    weekly = (sales.where(F.col("subcategory_nm") == top_sub)
              .groupBy("rgm_ppg", "period_id")
              .agg(F.sum("total_units").alias("units"),
                   F.sum("total_sales").alias("revenue"),
                   F.avg("any_promo_acv_pct").alias("promo"),
                   F.avg("acv_pct").alias("shelf")))
    pdf = weekly.toPandas()
    pdf["price"] = pdf.revenue / pdf.units
    return pdf.sort_values(["rgm_ppg", "period_id"])


def find_links(sales: DataFrame, settings) -> pd.DataFrame:
    """Returns one row per strong cross-price link: product_a, product_b, effect, p, relationship.

    The weekly aggregation is distributed (Spark); the pairwise fits run on the collected top-N
    series via the engine-free core, so the maths is identical to the notebook.
    """
    r = settings.relationships
    weekly = _weekly_top_subcategory(sales)
    return links_from_weekly(weekly,
                             top_n=int(r.compare_top_n_products),
                             min_weeks=int(r.minimum_shared_weeks),
                             alpha=float(r.significance_level),
                             min_effect=float(r.minimum_effect))
