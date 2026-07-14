"""Step 3 — group the products, name the groups, suggest a lever for each.  (PySpark + sklearn)

WHERE THIS RUNS, AND WHY.
Steps 1–2 are distributed in Spark because their cost scales with the number of *rows*
(84,550 today, millions in a full rollout). This step is different: it works on one row per
product — 353 rows, a few kilobytes — and that stays true at any realistic scale (a big
rollout is tens of thousands of products, still tiny). So the prepared feature matrix is
collected to the driver and the model is fit with scikit-learn.

That is a deliberate choice, not a shortcut:
  * There is no scale benefit to distributing a 353-row fit — the shuffle would cost more than
    the work.
  * The exact pipeline that makes these groups defensible — RobustScaler (median-based, so
    outliers don't drag it), PCA keeping 90% of variance, K-Means with 20 restarts — is the
    one validated in the analysis notebook. Reproducing it exactly keeps the pipeline's answer
    identical to the notebook's (same 3 groups, same 234/58/61 sizes). Spark MLlib has no
    RobustScaler and a different K-Means init, so it would quietly change the answer for no
    gain. The whole point of the pipeline is that it does NOT change the answer.

CHOOSING HOW MANY GROUPS.
The silhouette is highest at K=2 — but that answer puts 82% of the portfolio in one bucket:
technically clean, commercially useless. So the search starts at 3, looks at two scores rather
than one, and always checks the group sizes. That lands on K=3 (silhouette 0.41, CH 194).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import RobustScaler

from src._core import FEATURE_NAMES

# Extra product columns carried alongside the features for describing and publishing groups.
CARRY_COLS = ["rgm_ppg", "subcategory_nm", "brand_nm", "uom", "pack_size", "pack_count",
              "avg_units", "total_revenue"]

# Plain-language name -> (suggested lever, why). Named groups only; see name_groups() rules.
ACTIONS = {
    "Promo leaders":   ("Optimise the promo calendar", "Promo weeks clearly outsell quiet weeks"),
    "Steady core":     ("No single action", "Too big and too mixed for one lever"),
    "Shrinking shelf": ("Review product", "Investigate shelf loss"),
}


def to_pandas(features) -> pd.DataFrame:
    """Collect the one-row-per-product Spark table to the driver. Small by construction."""
    keep = [c for c in CARRY_COLS + FEATURE_NAMES]
    return features.select(*keep).toPandas()


def prepare(features_pdf: pd.DataFrame, settings):
    """Get the features onto a comparable footing before measuring distance between them.

    Clip the extreme 1% so a few spikes don't set the shape of the space; fill blanks with the
    median (a blank means "couldn't measure it", not zero); scale by the median/IQR so outliers
    don't drag the scaling; PCA to 90% of variance so features that overlap by construction
    aren't counted twice.
    """
    r = features_pdf[FEATURE_NAMES].copy()
    r = r.clip(r.quantile(0.01), r.quantile(0.99), axis=1)
    r = r.fillna(r.median())

    scaler = RobustScaler().fit(r)
    scaled = scaler.transform(r)
    pca = PCA(n_components=float(settings.grouping.keep_variation),
              random_state=int(settings.grouping.random_seed)).fit(scaled)
    return pca.transform(scaled), r, {"scaler": scaler, "pca": pca}


def choose_how_many(prepared, settings) -> tuple[int, pd.DataFrame]:
    low, high = settings.grouping.search_between
    seed, restarts = int(settings.grouping.random_seed), int(settings.grouping.restarts)
    scores = []
    for k in range(2, int(high) + 1):                       # start at 2 so the trap is visible
        labels = KMeans(n_clusters=k, n_init=restarts, random_state=seed).fit_predict(prepared)
        scores.append({"groups": k,
                       "silhouette": silhouette_score(prepared, labels),
                       "second_score": calinski_harabasz_score(prepared, labels),
                       "biggest_group": pd.Series(labels).value_counts(normalize=True).max()})
    scores = pd.DataFrame(scores)

    sensible = scores[(scores.groups >= int(low)) & (scores.groups <= int(high))].copy()
    sensible["rank"] = (sensible.silhouette.rank() + sensible.second_score.rank()) / 2
    best = int(sensible.sort_values(["rank", "silhouette"], ascending=False).iloc[0].groups)
    return best, scores


def fit(prepared, k, settings):
    return KMeans(n_clusters=int(k), n_init=int(settings.grouping.restarts),
                  random_state=int(settings.grouping.random_seed)).fit_predict(prepared)


def name_groups(profile: pd.DataFrame) -> dict:
    """Plain-language names from explicit rules on the group profile (z-scored across groups).

    Auditable in three lines: losing shops fast -> Shrinking shelf; heavily promoted ->
    Promo leaders; everything else -> Steady core. Rule order matters and is fixed.
    """
    z = (profile - profile.mean()) / profile.std(ddof=0)

    def one(g):
        r = z.loc[g]
        if r.shelf_trend <= -0.8:
            return "Shrinking shelf"
        if r.promo_activity >= 0.8:
            return "Promo leaders"
        return "Steady core"

    names, used = {}, {}
    for g in profile.index:
        base = one(g)
        used[base] = used.get(base, 0) + 1
        names[g] = base if used[base] == 1 else f"{base} {used[base]}"
    return names


def describe_groups(features_pdf: pd.DataFrame, labels):
    """Returns (profile of the 10 features by group, {group_id: name}, revenue% by group)."""
    with_groups = features_pdf.assign(group=labels)
    profile = with_groups.groupby("group")[FEATURE_NAMES].mean()
    names = name_groups(profile)
    revenue_pct = (with_groups.groupby("group").total_revenue.sum()
                   / with_groups.total_revenue.sum() * 100)
    return profile, names, revenue_pct


def suggest(features_pdf: pd.DataFrame, labels, names, revenue_pct) -> pd.DataFrame:
    """One readable lever per group. An if/then a commercial colleague can challenge."""
    sizes = pd.Series(labels).value_counts()
    rows = []
    for g, name in names.items():
        action, why = ACTIONS.get(name, ("Review by hand", "No rule matched this group"))
        rows.append({"group_name": name, "products": int(sizes[g]),
                     "revenue_pct": round(float(revenue_pct[g]), 1),
                     "action": action, "why": why})
    return (pd.DataFrame(rows).sort_values("revenue_pct", ascending=False)
            .reset_index(drop=True))


def split_biggest(prepared, features_pdf: pd.DataFrame, labels, settings) -> pd.DataFrame:
    """The biggest group is too coarse to act on, so split it again for targeting.

    On the real data 'Steady core' (234 products) splits into 3 layers: a big-seller backbone,
    a promo-seeking middle, and a micro-distribution tail stocked in ~2% of shops.
    """
    seed, restarts = int(settings.grouping.random_seed), int(settings.grouping.restarts)
    biggest = pd.Series(labels).value_counts().idxmax()
    idx = np.where(np.asarray(labels) == biggest)[0]

    best = (None, -1.0, None)
    for n in range(2, 6):
        sub = KMeans(n_clusters=n, n_init=restarts, random_state=seed).fit_predict(prepared[idx])
        score = silhouette_score(prepared[idx], sub)
        if score > best[1]:
            best = (n, score, sub)
    _, _, sub_labels = best
    return pd.DataFrame({"rgm_ppg": features_pdf.iloc[idx].rgm_ppg.values, "sub": sub_labels})
