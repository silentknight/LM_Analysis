"""Best polynomial approximation to -u*ln(u) on [0,1], by Remez exchange.

Used by the JVHW entropy estimator. The coefficients reach ~5e12 at degree 22,
so the exchange runs in `decimal` rather than float64; the Vandermonde solve is
hopeless in double precision at that degree.
"""

from decimal import Decimal as D
from decimal import getcontext
from functools import lru_cache
from math import cos, pi

getcontext().prec = 60
ZERO, ONE = D(0), D(1)


def _g(u):
    return ZERO if u <= 0 else -u * u.ln()


def _polyval(c, u):
    s, p = ZERO, u
    for ck in c:
        s += ck * p
        p *= u
    return s


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        inv = ONE / M[i][i]
        for r in range(i + 1, n):
            f = M[r][i] * inv
            if f:
                for col in range(i, n + 1):
                    M[r][col] -= f * M[i][col]
    x = [ZERO] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


@lru_cache(maxsize=32)
def best_poly(K: int, iters: int = 40, grid_mult: int = 30):
    """Coefficients of u^1..u^K and the sup error. Constant term is fixed at zero
    so that unobserved cells contribute nothing to the entropy estimate."""
    x = sorted(D(repr((cos(pi * (2 * i + 1) / (2 * (K + 1))) + 1) / 2)) for i in range(K + 1))
    x[0] = max(x[0], D("1e-9"))
    c = None
    for _ in range(iters):
        A = [[xi**k for k in range(1, K + 1)] + [D((-1) ** i)] for i, xi in enumerate(x)]
        sol = _solve(A, [_g(xi) for xi in x])
        c = sol[:K]

        def err(u):
            return _g(u) - _polyval(c, u)

        M = grid_mult * (K + 1)
        grid = [D(i) / M for i in range(M + 1)]
        vals = [err(u) for u in grid]
        ext = []
        for i in range(1, len(grid) - 1):
            if (vals[i] - vals[i - 1]) * (vals[i + 1] - vals[i]) <= 0:
                lo, hi = grid[i - 1], grid[i + 1]
                for _ in range(60):
                    m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                    lo, hi = (lo, m2) if abs(err(m1)) > abs(err(m2)) else (m1, hi)
                ext.append((lo + hi) / 2)
        ext = sorted(set([grid[1]] + ext + [ONE]))
        runs, cur = [], [ext[0]]
        for t in ext[1:]:
            if (err(t) > 0) == (err(cur[-1]) > 0):
                cur.append(t)
            else:
                runs.append(cur)
                cur = [t]
        runs.append(cur)
        ext = [max(r, key=lambda t: abs(err(t))) for r in runs]
        while len(ext) > K + 1:
            ext.pop(0 if abs(err(ext[0])) < abs(err(ext[-1])) else -1)
        if len(ext) < K + 1:
            break
        dev = [abs(err(t)) for t in ext]
        x = ext
        if max(dev) - min(dev) < max(dev) * D("1e-10"):
            break
    MM = 400 * (K + 1)
    emax = max(abs(_g(D(i) / MM) - _polyval(c, D(i) / MM)) for i in range(MM + 1))
    return tuple(c), emax
