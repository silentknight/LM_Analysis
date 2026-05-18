import numpy as np
from scipy import optimize


def powerlaw(x, amp, index):
    return amp * (x**index)


def fit_powerlaw(xdata, ydata, pinit=None, n_points=None):
    """Fit a power law in log-log space via least squares. Returns (amp, index, covar)."""
    if n_points is not None:
        xdata = xdata[:n_points]
        ydata = ydata[:n_points]
    if pinit is None:
        pinit = [0.0, -1.0]
    logx = np.log10(xdata.astype(float))
    logy = np.log10(ydata.astype(float))
    logyerr = np.ones_like(logy)

    def fitfunc(p, x):
        return p[0] + p[1] * x

    def errfunc(p, x, y, err):
        return (y - fitfunc(p, x)) / err

    out = optimize.leastsq(errfunc, pinit, args=(logx, logy, logyerr), full_output=1)
    pfinal, covar = out[0], out[1]
    index = pfinal[1]
    amp = 10.0 ** pfinal[0]
    return amp, index, covar
