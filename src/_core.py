"""Engine-free numeric core.

Every calculation that has to match the analysis notebook *exactly* lives here, in plain
numpy/pandas with no Spark import. The Spark modules (features.py, relationships.py) call these
functions inside `applyInPandas` / after a collect; the tests call them directly with no
SparkSession. One definition, two callers — so the distributed pipeline can never quietly
drift away from the notebook it's meant to reproduce.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

# The ten features the grouping actually sees.
FEATURE_NAMES = [
    "price_tier", "shelf_presence", "bumpiness", "price_moves", "growth",
    "seasonality", "shelf_trend", "promo_activity", "promo_share", "promo_lift",
]


def weekly_view(product_rows: pd.DataFrame) -> pd.DataFrame:
    """One product's weekly totals: units summed across retailers, shelf and promo averaged."""
    return (product_rows.groupby("period_id")
            .agg(units=("total_units", "sum"), shelf=("acv_pct", "mean"),
                 promo=("any_promo_acv_pct", "mean")).sort_index())


def movement(product_rows: pd.DataFrame, smoothing_weeks: int = 9,
             min_weeks_trend: int = 8, min_weeks_promo_lift: int = 26) -> dict:
    """Growth, shelf trend, seasonality and promo-lift for one product, on the sales rate."""
    w = weekly_view(product_rows)
    out = {"growth": np.nan, "shelf_trend": np.nan, "seasonality": 0.0, "promo_lift": np.nan}
    n = len(w)
    if n < min_weeks_trend:
        return out

    steady = w.shelf.rolling(smoothing_weeks, min_periods=4, center=True).median().fillna(w.shelf)
    x = np.arange(n, dtype=float)
    log_rate = np.log1p((w.units / steady.clip(lower=1)).values)

    slope, intercept = np.polyfit(x, log_rate, 1)
    out["growth"] = slope
    out["shelf_trend"] = np.polyfit(x, w.shelf.values, 1)[0]

    wobble = log_rate - (slope * x + intercept)
    cycle = 52 if n >= 104 else 13
    if n >= 2 * cycle and np.var(wobble) > 0:
        cyc = pd.Series(wobble).groupby(np.arange(n) % cycle).transform("mean")
        out["seasonality"] = max(0.0, 1 - np.var(wobble - cyc.values) / np.var(wobble))

    if n >= min_weeks_promo_lift and w.promo.std() > 0:
        heavy = (w.promo >= w.promo.quantile(0.75)).values
        quiet = (w.promo <= w.promo.quantile(0.25)).values
        if heavy.sum() >= 4 and quiet.sum() >= 4:
            out["promo_lift"] = float(np.expm1(np.median(wobble[heavy]) - np.median(wobble[quiet])))
    return out


def price_tier_within_group(product_level: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank of price-per-content within one (subcategory, uom) group."""
    content = (pd.to_numeric(product_level.pack_size, errors="coerce")
               * pd.to_numeric(product_level.pack_count, errors="coerce"))
    return product_level.assign(price_tier=(product_level.avg_price / content).rank(pct=True))


def links_from_weekly(weekly: pd.DataFrame, top_n: int = 40, min_weeks: int = 20,
                      alpha: float = 0.05, min_effect: float = 0.30) -> pd.DataFrame:
    """Cross-price links from a collected weekly product table (one subcategory).

    Regresses each top-N product's log weekly units on the OTHER product's log price, holding
    its own price, promo, shelf, a time trend and total category demand fixed. Sign of the
    other-price coefficient = compete (+) or sell together (-). Bonferroni + effect-size floor.
    """
    # Spark's toPandas() delivers DateType as plain datetime.date objects, whose Index has no
    # timedelta arithmetic (.days). Coerce to datetime64 so the time trend works from any caller.
    weekly = weekly.assign(period_id=pd.to_datetime(weekly.period_id))
    category_week = weekly.groupby("period_id").units.sum()
    top = weekly.groupby("rgm_ppg").units.sum().nlargest(top_n).index
    series = {p: g.set_index("period_id") for p, g in weekly[weekly.rgm_ppg.isin(top)].groupby("rgm_ppg")}

    pairs = []
    for a, b in itertools.combinations(top, 2):
        t = series[a].join(series[b], rsuffix="_other", how="inner").dropna()
        if len(t) < min_weeks:
            continue
        wk = np.asarray((t.index - t.index.min()).days, dtype=float) / 7.0
        cd = np.log(category_week.loc[t.index].values)
        X = np.column_stack([np.ones(len(t)), np.log(t.price_other), np.log(t.price),
                             t.promo / 100.0, np.log(t.shelf.clip(lower=0.5)),
                             wk - wk.mean(), cd - cd.mean()])
        beta, res, rank, _ = np.linalg.lstsq(X, np.log(t.units), rcond=None)
        dof = len(t) - rank
        if dof <= 0 or not res.size:
            continue
        se = np.sqrt(res[0] / dof * np.linalg.pinv(X.T @ X)[1, 1])
        if se == 0:
            continue
        pairs.append({"product_a": a, "product_b": b, "effect": float(beta[1]),
                      "p": float(2 * (1 - stats.t.cdf(abs(beta[1] / se), dof)))})

    links = pd.DataFrame(pairs)
    if links.empty:
        return links
    links = links.sort_values("p").reset_index(drop=True)
    links = links[(links.p < alpha / len(links)) & (links.effect.abs() >= min_effect)].copy()
    links["relationship"] = np.where(links.effect > 0, "compete", "sell together")
    return links.reset_index(drop=True)
