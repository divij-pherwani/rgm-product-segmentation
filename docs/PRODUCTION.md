# If this went to production

The analysis notebook is a one-off worked in pandas. `src/` is the same logic turned into
something that runs on its own, on **PySpark**. This note covers how it's shaped and — the part
worth arguing about — *what runs where, and why*.

The platform underneath matters less than the ideas: the same shape works on Databricks, EMR,
Glue, or a Spark job under Airflow.

---

## The shape of it

```
        data/ECON_POS.csv                 the raw weekly sales file
                 │
                 ▼
        ┌─────────────────┐
        │ 1. clean.py     │  Spark    drop placeholder rows, add barcodes together,
        │                 │           set aside products with too little history
        └─────────────────┘
                 │
                 ▼
        clean_sales.parquet              ← saved. Other analyses can start here.
                 │
                 ▼
        ┌─────────────────┐
        │ 2. features.py  │  Spark    one row per product: size, price, availability,
        │                 │  + UDF    growth, seasonality, promo response
        └─────────────────┘
                 │
                 ▼
        features.parquet                 ← saved, with a run date attached
                 │
                 ▼
        ┌─────────────────┐
        │ 3. group.py     │  driver   prepare, pick the number of groups, fit,
        │                 │  sklearn  name the groups, suggest a lever, split the biggest
        └─────────────────┘
                 │
                 ▼
        groups.csv + suggestions.csv     ← what the business actually looks at
                 │
                 ▼
        ┌─────────────────┐
        │ 4. check.py     │  driver   does it hold up on resampling? did groups move?
        │                 │           → PUBLISH, or STOP and page someone
        └─────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ 5. relationships│  Spark    weekly aggregation distributed; the pairwise
        │      .py        │  + driver cross-price fits on the top-N sellers
        └─────────────────┘
```

`notebooks/00_run_pipeline.ipynb` runs all five steps in order, reading `config.yaml` and calling
`src/`. It's the closest thing here to a real scheduled run, and it ends by checking the pipeline
gives the same answer as the analysis notebook.

---

## What runs where, and why

This is the decision a reviewer should push on, so here it is in the open.

**Steps 1–2 are distributed in Spark.** Their cost scales with the number of *rows* — 84,550 on
the recruitment data, but millions across more categories, retailers and years. Cleaning is a
straight pass over the rows (filter, then group-and-aggregate); feature-building is one independent
calculation per product. Both are exactly what Spark is for.

The per-product time-series metrics (growth, seasonality, promo-lift) run through
`groupBy("rgm_ppg").applyInPandas(...)`. Each product is independent of every other, so Spark ships
one product to one core and runs the *same numpy the notebook uses* on it — embarrassingly
parallel, and numerically identical to the pandas version because it **is** the pandas version.

**Steps 3–5 (the model) run on the driver, with scikit-learn.** They work on one row per product
— 353 rows here, and still only tens of thousands after a big rollout. That is tiny at any
realistic scale, so distributing it would add shuffle cost for no speed-up. More importantly, the
exact pipeline that makes the groups defensible — `RobustScaler` (median-based, so a few spikes
don't set the scale), `PCA` keeping 90% of variance, `KMeans` with 20 restarts — is the one
validated in the analysis. Spark MLlib has no RobustScaler and a different K-Means initialisation,
so switching to it would quietly change the answer **and buy nothing**, because the input is 353
rows either way. So the small model stays on the exact library the notebook used, and the pipeline
reproduces the notebook to the digit (same 3 groups, same 234 / 58 / 61 sizes).

That split — **Spark for the row-scale ETL and feature engineering, single-node for the small
aggregated model** — is a deliberate, standard pattern, not a shortcut. The numeric core that has
to match the notebook lives in one engine-free module (`src/_core.py`) that both the Spark UDFs and
the tests call, so the two can never silently drift.

The one part that's inherently all-against-all is the cross-price step: comparing pairs of products
is O(n²). Its weekly aggregation is distributed, but the pairwise regressions run on the collected
top-N series — bounded to ~40 products, that's 780 tiny fits.

---

## The four things that actually matter

### 1. Save each step, don't recompute it
Every stage writes its output (`parquet` for the big intermediate tables). If step 3 fails you
don't redo 1 and 2; `clean_sales.parquet` is useful to other people; and if a final number looks
wrong you can go and look at exactly what fed into it.

### 2. The checking step can stop the pipeline
`check.py` re-runs the grouping on random samples. If the groups don't hold up, it does **not**
publish the new ones — the old groups stay live and it pages someone. People build plans on these
groups; a segmentation that quietly changes underneath them is worse than one that's a bit stale.
On the data it comes back at 0.96, well over the 0.60 bar set in advance.

### 3. Track how many products change group
Every run, compare against the last (`check.how_many_products_moved`). If a big chunk of the
portfolio has moved (start at 20%), that's not a routine refresh — that's the map being redrawn,
and people need to be told. Section 5.3 of the notebook shows why this matters: promo activity
moved 6.2 ACV points over the three years, and whatever caused that will cause it again.

### 4. Save the fitted model, don't refit it
When a new product needs a group, score it against the *existing* groups rather than triggering a
whole new grouping — otherwise the groups drift every time anything happens and nobody can tell
whether a product moved because it changed or because the model did. `group.prepare` returns the
fitted scaler and PCA; persist those plus the K-Means centres. That's also how the 13 products set
aside for too little history get back in, once they have a quarter of data.

---

## Where it would slow down

| Step | What happens as the data grows |
|---|---|
| **1. Cleaning** | Scales with rows. Distributed already — this is the part Spark earns its keep on. |
| **2. Features** | Scales with rows for the aggregates, and with *products* for the per-product UDF — but each product is independent, so it parallelises perfectly across the cluster. |
| 3. Grouping | Barely changes. One row per product: 353 today, maybe 30,000 in a big rollout. Still small; stays on the driver. |
| 5. Relationships | **The dangerous one.** Pairwise is O(n²): double the products and you quadruple the work. Held in check by only comparing within a subcategory and only the top-N sellers. Without those limits it falls over on a big category. |

The useful thing about that table: the *modelling* isn't the hard part. The hard part is the data
work before it, and the pairwise step after. That's usually the way round.

---

## Folder layout

```
Product-Segmentation/
├── config.yaml            every setting, in one file
├── requirements.txt       pyspark for the pipeline; pandas/sklearn for the model + notebook
├── data/                  the raw file (not committed)
├── notebooks/
│   ├── product_segmentation.ipynb   the analysis (pandas)
│   └── 00_run_pipeline.ipynb        runs the five steps below, in Spark
├── src/
│   ├── _core.py           engine-free numeric core — shared by Spark and the tests
│   ├── config.py          reads config.yaml, complains if something's missing
│   ├── clean.py           step 1  (Spark)
│   ├── features.py        step 2  (Spark + applyInPandas)
│   ├── group.py           step 3  (driver / sklearn)
│   ├── check.py           step 4  (driver)
│   └── relationships.py   step 5  (Spark + driver)
├── tests/
│   └── test_pipeline.py   plant patterns with a known answer; check we find them
├── output/                groups, suggestions, checks
└── docs/                  this note, the report, and the figures
```

---

## Testing something with no right answer

Clustering has nothing to be "accurate" against, so what do you test? **Make up products where you
know the answer.** Generate a fake product that's deliberately seasonal, another deliberately
growing, another that's a deliberate substitute for a second product. Then check the features and
the grouping find what you planted. If the code can't spot a pattern put there on purpose, it
shouldn't be trusted on real data.

`tests/test_pipeline.py` does this in two layers. The numeric core and the model are tested
directly in pandas/sklearn — no Spark needed, so `pytest` is fast on a laptop. The two Spark steps
(`clean`, `features`) are then tested end-to-end against a pandas twin of the same aggregation, to
prove the distributed version matches; those tests skip automatically where PySpark isn't
installed. The suite also checks that things get *rejected* properly — an unstable grouping
shouldn't get published.
