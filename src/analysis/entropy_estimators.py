from abc import ABC, abstractmethod
from decimal import Decimal as D
from math import ceil, log

import numpy as np
import scipy.special as spec

from .polyapprox import best_poly


class BaseEstimator(ABC):
    @abstractmethod
    def entropy(self, counts: np.ndarray, log_base_val: float) -> float:
        """Entropy of a discrete distribution from observed counts. Result in log_base_val units."""

    def probabilities(self, counts: np.ndarray) -> np.ndarray:
        """Probability estimate for each observed symbol. Defaults to MLE."""
        return counts / np.sum(counts)


class StandardEstimator(BaseEstimator):
    def entropy(self, counts, log_base_val):
        p = counts / np.sum(counts)
        return -np.sum(p * np.log(p)) / log_base_val


class MillerMadowEstimator(BaseEstimator):
    """MLE + (S̃-1)/(2N) bias correction (Miller 1955)."""

    def entropy(self, counts, log_base_val):
        N = np.sum(counts)
        p = counts / N
        return (-np.sum(p * np.log(p)) + (len(counts) - 1) / (2 * N)) / log_base_val


class GrassbergerEstimator(BaseEstimator):
    """Grassberger (1988): H = log(N) - (1/N)·Σ nᵢ·ψ(nᵢ)."""

    def entropy(self, counts, log_base_val):
        N = np.sum(counts)
        return (np.log(N) - np.sum(counts * spec.digamma(counts)) / N) / log_base_val


class GrassbergerAltEstimator(BaseEstimator):
    """Grassberger (1988) Eq. 13: ψ(n) + (-1)^n/(n(n+1)) removes parity artifacts."""

    @staticmethod
    def _g(c):
        sign = np.where(c.astype(np.int64) % 2 == 0, 1.0, -1.0)
        return spec.digamma(c) + sign / (c * (c + 1))

    def entropy(self, counts, log_base_val):
        N = np.sum(counts)
        return (np.log(N) - np.sum(counts * self._g(counts)) / N) / log_base_val


class ChaoShenEstimator(BaseEstimator):
    """Chao & Shen (2003): Good-Turing coverage + Horvitz-Thompson correction for unseen events."""

    def probabilities(self, counts):
        N = np.sum(counts)
        f1 = np.sum(counts == 1)
        C = max(1.0 - f1 / N, 1.0 / N)
        return C * counts / N

    def entropy(self, counts, log_base_val):
        N = np.sum(counts)
        p_hat = self.probabilities(counts)
        # log1p(-1.0) = -inf when p_hat=1 (single-symbol edge case); suppress the warning
        # since exp(-inf)=0 and ht_denom=1 is the correct limit.
        with np.errstate(divide="ignore"):
            ht_denom = np.maximum(1.0 - np.exp(N * np.log1p(-p_hat)), 1e-15)
        return -np.sum(p_hat / ht_denom * np.log(p_hat)) / log_base_val


class JVHWEstimator(BaseEstimator):
    """Jiao et al. (2015) / Wu and Yang (2016): unbiased estimator of the best
    polynomial approximation to -p*ln(p) where the functional is non-smooth, and
    a bias-corrected plug-in elsewhere. Attains the n ~ S/ln(S) minimax rate.

    The constant term of the approximation is fixed at zero, so unobserved cells
    contribute nothing and the alphabet size need not be supplied.
    """

    def __init__(self, c1: float = 1.6, c2: float = 1.2):
        self.c1, self.c2 = c1, c2

    def entropy(self, counts, log_base_val):
        counts = np.asarray(counts)
        N = int(np.sum(counts))
        if N <= 1:
            return 0.0
        # Symbols with p_hat <= delta take the polynomial branch; the
        # approximation is built on [0, 2*delta] so that those symbols sit in the
        # interior of its domain. Classifying at the domain edge instead puts the
        # bulk of a well-sampled alphabet exactly where the branch is worst, and
        # drives the estimate far below the truth.
        delta = min(self.c1 * log(N) / N, 0.5)
        thresh = 2.0 * delta
        m_max = int(delta * N)
        # The unbiased estimator of p^k uses the falling factorial (m)_k, whose
        # variance explodes once k approaches the largest count in the branch.
        K = min(4 + ceil(self.c2 * log(N)), 22, max(m_max, 1))
        c, _ = best_poly(K)

        # a_k on [0, thresh] from the coefficients on [0, 1]
        t, n = D(repr(thresh)), D(N)
        a = [c[k - 1] * t ** (1 - k) for k in range(1, K + 1)]
        a[0] -= t.ln()

        small = counts <= m_max
        total = D(0)
        if np.any(small):
            vals, reps = np.unique(counts[small], return_counts=True)
            for m, rep in zip(vals.astype(np.int64).tolist(), reps.tolist()):
                acc, ff = D(0), D(1)
                for k in range(1, min(K, m) + 1):
                    ff *= D(m - k + 1) / n
                    acc += a[k - 1] * ff
                total += acc * D(rep)
        H = float(total)
        if np.any(~small):
            p = counts[~small] / N
            H += float(np.sum(-p * np.log(p) + 1.0 / (2 * N)))
        return H / log_base_val


class NSBEstimator(BaseEstimator):
    """Nemenman-Shafee-Bialek Bayesian estimator. Requires: pip install ndd."""

    def __init__(self, k: int | None = None):
        self.k = k  # full alphabet size; if None, ndd infers from observed counts (underestimates)

    def entropy(self, counts, log_base_val):
        try:
            import ndd
        except ImportError:
            raise ImportError("NSB estimator requires: pip install ndd")
        kwargs = {"k": self.k} if self.k is not None else {}
        return ndd.entropy(counts.astype(int), **kwargs) / log_base_val


# fmt: off
_REGISTRY: dict[str, type[BaseEstimator]] = {
    "standard":         StandardEstimator,
    "miller_madow":     MillerMadowEstimator,
    "grassberger":      GrassbergerEstimator,
    "grassberger_alt":  GrassbergerAltEstimator,
    "chao_shen":        ChaoShenEstimator,
    "jvhw":             JVHWEstimator,
    "nsb":              NSBEstimator,
}
# fmt: on

# Kept because experiment directories on disk are named after the old key; the
# estimator is Grassberger 1988 Eq. 13, not the 2003 integral form.
_ALIASES = {"grassberger_2003": "grassberger_alt"}

ESTIMATOR_NAMES = list(_REGISTRY)


def get_estimator(name: str) -> BaseEstimator:
    name = _ALIASES.get(name, name)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown estimator {name!r}. Choose from: {ESTIMATOR_NAMES}")
    return _REGISTRY[name]()
