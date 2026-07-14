# Output

Everything in here is written by the notebooks. Don't edit these by hand — re-run instead.

Two things write here. The **pipeline** (`notebooks/00_run_pipeline.ipynb`, PySpark) writes the
business-facing set and overwrites those files in place. The **analysis** notebook
(`notebooks/product_segmentation.ipynb`, pandas) wipes the folder at the start of its run and
writes a slightly richer, more exploratory set. Where the two write a file of the same name, the
*numbers* agree — that's the point, and the pipeline checks it — though the pipeline's versions
carry a couple of extra convenience columns (e.g. a `relationship` label on the cross-price links).

## Written by the pipeline (the business set)

| File | Rows | What it is |
|---|---|---|
| `groups.csv` | 353 | Every product with its group id and name, plus the 10 features. The table the business uses. |
| `group_profiles.csv` | 3 | What each group looks like on average across the 10 features. |
| `suggestions.csv` | 3 | The lever for each group, and why. Steady core gets "no single action" — it's split instead. |
| `products_set_aside.csv` | 13 | Products with under 13 weeks of history — too new to judge. A watch-list, not a bin. |
| `biggest_group_split.csv` | 234 | The largest group (Steady core) split again into 3 layers (86 / 33 / 115). |
| `cross_price_links.csv` | 45 | Which products compete (28) and which sell together (17). One row per link. |

## Also written by the analysis notebook

| File | Rows | What it is |
|---|---|---|
| `product_groups.csv` | 353 | Same as `groups.csv` but with the catalogue fields (format, pack size…) attached. |
| `cross_price_node_lookup.csv` | 33 | The products that appear in the cross-price network, with a short id and their link count. |
| `retailer_profiles.csv` | 6 | Each retailer's revenue, range, shelf presence and promo intensity. |

## The three groups

| Group | Products | Revenue | Suggestion |
|---|---|---|---|
| Promo leaders | 58 | 41% | Optimise the promo calendar — promo weeks clearly outsell quiet weeks |
| Steady core | 234 | 49% | No single action — too big and mixed for one lever, so it's split (86 / 33 / 115) |
| Shrinking shelf | 61 | 10% | Review product — losing shelf fast, while the rate where stocked holds up |
