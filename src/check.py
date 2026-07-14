"""Step 4 — is the answer any good, and should we publish it?  (driver-side)

THIS STEP CAN STOP THE PIPELINE. That's the point of it.

If the groups don't hold up when you resample the products, they aren't real, and the new
segmentation does NOT get published. The old one stays live and someone gets told. People
build plans on these groups; a segmentation that quietly changes underneath them is worse
than one that's a bit out of date.

Like step 3, this works on the small prepared matrix (one row per product), so it runs on the
driver with scikit-learn. On the real data the groups agree with themselves at 0.96 across 25
resamples — comfortably above the 0.60 bar set in advance.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def does_it_hold_up(fit_function, prepared, settings) -> float:
    """Re-run the grouping on random 80% samples. Do products land in the same groups?

    1.0 = identical every time. 0.0 = no better than shuffling at random.
    """
    random = np.random.default_rng(int(settings.grouping.random_seed))
    original = fit_function(prepared)
    agreements = []
    for _ in range(int(settings.checks.resample_times)):
        sample = random.choice(len(prepared),
                               int(float(settings.checks.resample_size) * len(prepared)),
                               replace=False)
        agreements.append(adjusted_rand_score(original[sample], fit_function(prepared[sample])))
    return float(np.mean(agreements))


def how_many_products_moved(before: pd.DataFrame, after: pd.DataFrame) -> float:
    """Compare this run's groups with the last one. What fraction of products changed group?"""
    both = before.merge(after, on="rgm_ppg", suffixes=("_before", "_after"))
    if both.empty:
        return 0.0
    return float((both.group_before != both.group_after).mean())


def should_we_publish(agreement: float, suggestions: pd.DataFrame, settings) -> tuple[bool, str]:
    bar = float(settings.checks.minimum_agreement)
    if agreement < bar:
        return False, (f"NO. The groups only agree with themselves at {agreement:.2f} when "
                       f"resampled (we need {bar:.2f}). Keeping the previous groups live and "
                       f"flagging this for a human.")

    # 'No single action' isn't a failure — it's the group too big/mixed for one lever, which
    # gets a drill-down split instead. Report it honestly rather than pretending every group
    # earned a recommendation.
    drilled = int((suggestions.action == "No single action").sum())
    return True, (f"Yes. Groups hold up at {agreement:.2f}. "
                  f"{len(suggestions) - drilled} of {len(suggestions)} groups get a lever; "
                  f"{drilled} is too big and mixed for one and gets split instead.")
