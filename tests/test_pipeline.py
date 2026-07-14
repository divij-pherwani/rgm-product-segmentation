"""Testing something that has no right answer.

Clustering has nothing to be "accurate" against — nobody told us the correct groups — so what
do you test? The answer here is: make up products where I know the answer (deliberately
seasonal, deliberately growing, deliberately price-competitive), and check the code finds what
I planted. If it can't spot a pattern I put there on purpose, I shouldn't trust it on real data.

Two layers:
  * The numeric core (src/_core.py) and the model (src/group.py, src/check.py) are pure
    numpy/pandas/sklearn and are tested directly — no SparkSession needed.
  * The Spark steps (src/clean.py, src/features.py) are tested end-to-end against a pandas twin
    of the same aggregation. Those tests skip automatically where pyspark isn't installed, so
    `pytest` is green on a laptop and thorough on a cluster.

Run with:  pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import _core, check, config, group

SETTINGS = config.load(ROOT / "config.yaml")


# --------------------------------------------------------------------------------------------
# A fake product that behaves exactly how I tell it to.
# --------------------------------------------------------------------------------------------
def make_a_product(name, weekly_units, price, price_sensitivity, promo_often,
                   growth_rate, seasonal_swing, weeks=104, seed=0):
    random = np.random.default_rng(seed)
    rows = []
    for shop in ["shop_A", "shop_B", "shop_C"]:
        shop_size = random.uniform(0.6, 1.4)
        for week in range(weeks):
            on_promo = random.random() < promo_often
            week_price = price * (0.75 if on_promo else 1.0) * random.normal(1, 0.02)
            demand = (np.log(weekly_units * shop_size)
                      + price_sensitivity * np.log(week_price / price)
                      + growth_rate * week
                      + seasonal_swing * np.sin(2 * np.pi * week / 52)
                      + (0.4 if on_promo else 0))
            units = max(1, int(np.exp(demand + random.normal(0, 0.1))))
            rows.append(dict(
                rgm_ppg=name, retailer_nm=shop, subcategory_nm="TEST", brand_nm="TEST", uom="CT",
                product_unit_size=10, product_pack_count=1, product_id=f"{name}_barcode",
                period_id=(pd.Timestamp("2024-01-07") + pd.Timedelta(weeks=week)).strftime("%m/%d/%Y"),
                total_units=units, total_volume=units * 10, total_sales=units * week_price,
                acv_pct=random.uniform(60, 90),
                any_promo_acv_pct=random.uniform(20, 60) if on_promo else 0.0,
                acv_tpr_only=random.uniform(20, 60) if on_promo else 0.0,
                feature_and_display_acv_pct=0.0, feature_without_display_acv_pct=0.0,
                display_without_feature_acv_pct=0.0,
                any_promo_units=units * 0.5 if on_promo else 0.0,
                any_promo_amt=units * week_price * 0.5 if on_promo else 0.0))
    return pd.DataFrame(rows)


def clean_features_pandas(raw: pd.DataFrame, settings) -> pd.DataFrame:
    """Pandas twin of src/clean.run + src/features.run — the oracle for the Spark parity tests
    and a convenient driver for the model tests. Mirrors the Spark aggregation exactly."""
    raw = raw.copy()
    raw["period_id"] = pd.to_datetime(raw.period_id, format="%m/%d/%Y")
    SALES = ["total_units", "total_volume", "total_sales", "any_promo_units", "any_promo_amt"]
    ACV = ["acv_pct", "any_promo_acv_pct", "acv_tpr_only", "feature_and_display_acv_pct",
           "feature_without_display_acv_pct", "display_without_feature_acv_pct"]
    META = ["subcategory_nm", "brand_nm", "uom", "product_unit_size", "product_pack_count"]
    floor = float(settings.cleaning.minimum_price_per_pack)
    minw = int(settings.cleaning.minimum_weeks_of_history)
    ppp = raw.total_sales / raw.total_units.replace(0, np.nan)
    kept = raw[(ppp >= floor) & (raw.total_units > 0)]
    meta = kept.groupby("rgm_ppg")[META].first()
    agg = (kept.groupby(["rgm_ppg", "retailer_nm", "period_id"], as_index=False)
           .agg({**{c: "sum" for c in SALES}, **{c: "max" for c in ACV}}).merge(meta, on="rgm_ppg"))
    agg["price_per_pack"] = agg.total_sales / agg.total_units
    weeks = agg.groupby("rgm_ppg").period_id.nunique()
    sales = agg[~agg.rgm_ppg.isin(weeks[weeks < minw].index)].reset_index(drop=True)

    prod = (sales.groupby(["rgm_ppg", "subcategory_nm", "brand_nm", "uom"], as_index=False)
            .agg(avg_units=("total_units", "mean"), unit_swing=("total_units", "std"),
                 avg_price=("price_per_pack", "mean"), price_swing=("price_per_pack", "std"),
                 total_revenue=("total_sales", "sum"), shelf_presence=("acv_pct", "mean"),
                 promo_revenue=("any_promo_amt", "sum"), promo_activity=("any_promo_acv_pct", "mean"),
                 pack_size=("product_unit_size", "first"), pack_count=("product_pack_count", "first"),
                 n_weeks=("period_id", "nunique")))
    prod["bumpiness"] = prod.unit_swing / prod.avg_units
    prod["price_moves"] = prod.price_swing / prod.avg_price
    prod["promo_share"] = prod.promo_revenue / prod.total_revenue
    content = pd.to_numeric(prod.pack_size, errors="coerce") * pd.to_numeric(prod.pack_count, errors="coerce")
    prod["price_tier"] = (prod.avg_price / content).groupby([prod.subcategory_nm, prod.uom]).rank(pct=True)
    moves = pd.DataFrame([{"rgm_ppg": p, **_core.movement(
        g, int(settings.features.shelf_smoothing_weeks),
        int(settings.features.minimum_weeks_for_trend),
        int(settings.features.minimum_weeks_for_promo_lift))} for p, g in sales.groupby("rgm_ppg")])
    return prod.merge(moves, on="rgm_ppg")


# --------------------------------------------------------------------------------------------
# Numeric core — no Spark needed.
# --------------------------------------------------------------------------------------------
def test_growth_is_picked_up_in_the_right_direction():
    growing = make_a_product("GROW", 500, 3.0, -1.0, 0.1, growth_rate=+0.004, seasonal_swing=0.05)
    shrinking = make_a_product("SHRINK", 500, 3.0, -1.0, 0.1, growth_rate=-0.004,
                               seasonal_swing=0.05, seed=2)
    g = _core.movement(growing.assign(period_id=pd.to_datetime(growing.period_id, format="%m/%d/%Y")))
    s = _core.movement(shrinking.assign(period_id=pd.to_datetime(shrinking.period_id, format="%m/%d/%Y")))
    assert g["growth"] > 0 > s["growth"]


def test_a_seasonal_product_scores_higher_than_a_flat_one():
    seasonal = make_a_product("SEAS", 800, 4.0, -1.0, 0.1, 0.0, seasonal_swing=0.9)
    flat = make_a_product("FLAT", 800, 4.0, -1.0, 0.1, 0.0, seasonal_swing=0.02, seed=1)
    a = _core.movement(seasonal.assign(period_id=pd.to_datetime(seasonal.period_id, format="%m/%d/%Y")))
    b = _core.movement(flat.assign(period_id=pd.to_datetime(flat.period_id, format="%m/%d/%Y")))
    assert a["seasonality"] > b["seasonality"]


def test_promo_lift_is_blank_when_promotions_never_vary():
    never = make_a_product("NOPROMO", 500, 3.0, -1.0, promo_often=0.0, growth_rate=0.0,
                           seasonal_swing=0.05)
    m = _core.movement(never.assign(period_id=pd.to_datetime(never.period_id, format="%m/%d/%Y")))
    assert np.isnan(m["promo_lift"])


def test_price_tier_ranks_within_its_group():
    df = pd.DataFrame({"avg_price": [1.0, 2.0, 3.0], "pack_size": [1, 1, 1],
                       "pack_count": [1, 1, 1]})
    out = _core.price_tier_within_group(df)
    assert list(out.price_tier.rank()) == [1.0, 2.0, 3.0]  # cheapest lowest tier


def test_the_naming_rules_are_auditable_and_correct():
    # A profile with one clearly shrinking group and one clearly heavily-promoted group.
    profile = pd.DataFrame(index=[0, 1, 2], columns=_core.FEATURE_NAMES, data=0.0)
    profile.loc[0, "shelf_trend"] = -3.0     # losing shops fast -> Shrinking shelf
    profile.loc[1, "promo_activity"] = 30.0  # heavily promoted   -> Promo leaders
    names = group.name_groups(profile)
    assert names[0] == "Shrinking shelf" and names[1] == "Promo leaders" and names[2] == "Steady core"


def test_the_whole_grouping_finds_the_kinds_of_product_i_planted():
    from sklearn.metrics import adjusted_rand_score
    everything = pd.concat([
        *[make_a_product(f"EXP_{i}", 300, 8.0, -0.6, 0.05, 0.001, 0.05, seed=i) for i in range(8)],
        *[make_a_product(f"PROMO_{i}", 3000, 2.0, -2.5, 0.45, -0.001, 0.05, seed=10 + i) for i in range(8)],
        *[make_a_product(f"SEAS_{i}", 800, 4.0, -1.2, 0.15, 0.004, 0.9, seed=20 + i) for i in range(8)],
    ])
    truth = {n: (0 if n.startswith("EXP") else 1 if n.startswith("PROMO") else 2)
             for n in everything.rgm_ppg.unique()}
    feat = clean_features_pandas(everything, SETTINGS)
    prepared, _, _ = group.prepare(feat, SETTINGS)
    labels = group.fit(prepared, 3, SETTINGS)
    agreement = adjusted_rand_score(feat.rgm_ppg.map(truth), labels)
    assert agreement > 0.7, f"only found the planted groups at {agreement:.2f}"


def test_an_unstable_grouping_does_not_get_published():
    suggestions = pd.DataFrame({"action": ["Optimise the promo calendar", "Review product"]})
    publish, reason = check.should_we_publish(0.30, suggestions, SETTINGS)   # below the bar
    assert publish is False and "previous groups" in reason


def test_a_stable_grouping_does_get_published():
    suggestions = pd.DataFrame({"action": ["Optimise the promo calendar", "No single action",
                                           "Review product"]})
    publish, reason = check.should_we_publish(0.96, suggestions, SETTINGS)
    assert publish is True and "hold up" in reason


def test_cross_price_recovers_a_planted_substitute_pair():
    # Two products where A's sales rise when B is expensive (a substitute -> "compete").
    # shelf/promo vary week to week, like the real data, so no degenerate columns.
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2024-01-07", periods=60, freq="7D")
    pa = 3 + rng.normal(0, 0.3, len(weeks))
    pb = 3 + rng.normal(0, 0.3, len(weeks))
    ua = np.exp(4.0 + 0.9 * np.log(pb) - 0.5 * np.log(pa) + rng.normal(0, 0.05, len(weeks)))
    ub = np.exp(4.0 - 0.5 * np.log(pb) + rng.normal(0, 0.05, len(weeks)))
    rows = []
    for name, price, units in [("A", pa, ua), ("B", pb, ub)]:
        for wk, pr, u in zip(weeks, price, units):
            rows.append(dict(rgm_ppg=name, period_id=wk, units=float(u), price=float(pr),
                             promo=float(rng.uniform(0, 20)), shelf=float(rng.uniform(60, 80))))
    links = _core.links_from_weekly(pd.DataFrame(rows), top_n=2, min_weeks=20,
                                    alpha=0.05, min_effect=0.30)
    assert not links.empty and (links.relationship == "compete").any()


# --------------------------------------------------------------------------------------------
# Spark integration — skipped automatically where pyspark isn't installed.
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    s = (SparkSession.builder.master("local[2]").appName("tests")
         .config("spark.sql.shuffle.partitions", "4").getOrCreate())
    yield s
    s.stop()


def _planted_raw():
    return pd.concat([
        make_a_product("EXP", 300, 8.0, -0.6, 0.05, 0.001, 0.05, seed=1),
        make_a_product("PROMO", 3000, 2.0, -2.5, 0.45, -0.001, 0.05, seed=2),
        make_a_product("SEAS", 800, 4.0, -1.2, 0.15, 0.004, 0.9, seed=3),
    ])


def test_spark_clean_matches_the_pandas_twin(spark):
    from src import clean
    raw = _planted_raw()
    sdf = spark.createDataFrame(raw)
    sales, set_aside = clean.run(sdf, SETTINGS)
    got = sales.count()
    twin = clean_features_pandas(raw, SETTINGS)          # products surviving cleaning
    assert sales.select("rgm_ppg").distinct().count() == len(twin)
    assert got > 0 and set_aside == []


def test_spark_features_match_the_pandas_twin(spark):
    from src import clean, features
    raw = _planted_raw()
    sales, _ = clean.run(spark.createDataFrame(raw), SETTINGS)
    spark_feat = features.run(sales, SETTINGS).toPandas().set_index("rgm_ppg").sort_index()
    twin = clean_features_pandas(raw, SETTINGS).set_index("rgm_ppg").sort_index()
    for col in _core.FEATURE_NAMES:
        a, b = spark_feat[col].astype(float), twin[col].astype(float)
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)
