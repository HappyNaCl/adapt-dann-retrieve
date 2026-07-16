"""Significance testing fixed in Phase 0: paired t-test + permutation test.

Both operate on per-query metric values from two systems over the same queries
(paired by information need). Holm-Bonferroni correction for multiple metrics.
"""
import numpy as np
from scipy import stats


def paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Returns (t_statistic, p_value) for paired scores a vs b."""
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 100_000, seed: int = 42) -> float:
    """Two-sided paired randomization test on the mean difference."""
    rng = np.random.default_rng(seed)
    diff = a - b
    observed = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.shape[0]))
    perm_means = np.abs((signs * diff).mean(axis=1))
    return float((perm_means >= observed).mean())


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Returns {name: significant?} under Holm-Bonferroni correction."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    significant: dict[str, bool] = {}
    rejected_so_far = True
    for i, (name, p) in enumerate(items):
        rejected_so_far = rejected_so_far and p <= alpha / (m - i)
        significant[name] = rejected_so_far
    return significant
