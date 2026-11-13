"""Two-way clustered bootstrap: clinicians and images are resampled independently, and a dialogue's weight is the
product of its clinician's and its image's counts.
"""

import numpy as np


def cluster_bootstrap(values, clinicians, images, stat=None, n_boot=2000, seed=0, alpha=0.05):
    """Percentile interval of a weighted mean (or of `stat(values, weights)`). Returns (lo, hi).

    A replicate in which every weight is zero is drawn again. With one clinician and one image the interval collapses
    to the point estimate.
    """
    values = np.asarray(values, float)
    if not len(values):
        raise ValueError("no values")
    cu, ci = np.unique(np.asarray(clinicians, dtype=object).astype(str), return_inverse=True)
    iu, ii = np.unique(np.asarray(images, dtype=object).astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    stat = stat or (lambda v, w: float((v * w).sum() / w.sum()))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        while True:
            w = (
                rng.multinomial(len(cu), np.full(len(cu), 1 / len(cu)))[ci]
                * rng.multinomial(len(iu), np.full(len(iu), 1 / len(iu)))[ii]
            )
            if w.sum() > 0:
                break
        reps[b] = stat(values, w.astype(float))
    return float(np.quantile(reps, alpha / 2)), float(np.quantile(reps, 1 - alpha / 2))
