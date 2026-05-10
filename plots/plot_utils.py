import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

SEABORN_STYLE = 'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn'


def powerlaw(x, amp, index):
    return amp * (x ** index)


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
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy, logyerr), full_output=1)
    pfinal, covar = out[0], out[1]
    index = pfinal[1]
    amp = 10.0 ** pfinal[0]
    return amp, index, covar


def apply_plot_style(ax, xlabel, ylabel, legend_loc='upper right', legend_ncol=1):
    """Apply standard grid, labels, and legend to an axes object."""
    ax.tick_params(labelsize='large', width=5)
    ax.grid(True)
    ax.grid(which='major', linestyle='-.', linewidth='0.5', color='grey')
    ax.grid(which='minor', linestyle=':', linewidth='0.2', color='grey')
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    lgd = ax.legend(loc=legend_loc, shadow=True, fancybox=True,
                    ncol=legend_ncol, numpoints=1, prop={'size': 12})
    return lgd


def group_output_name(group):
    """Derive the base output filename for a group of any shape.

    Strategy (in order):
    1. Single item → use the name as-is.
    2. Multiple items with a common underscore-component prefix → use that prefix.
    3. No common prefix but a test split exists → use that split's base name.
    4. Fallback → first item.
    """
    if len(group) == 1:
        return group[0]

    # Longest common underscore-component prefix
    parts_list = [name.split('_') for name in group]
    common = []
    for components in zip(*parts_list):
        if len(set(components)) == 1:
            common.append(components[0])
        else:
            break
    if common:
        return '_'.join(common)

    # No common prefix: use the first test split's base name (strips the test suffix)
    for name in group:
        if name.rsplit('_', 1)[-1].startswith('test'):
            parts = name.split('_')
            return '_'.join(parts[:-1]) if len(parts) > 1 else name

    return group[0]


def label_for(name, group):
    """Legend label relative to the group's common prefix.

    If the name starts with the group output name, return only the suffix
    (e.g. 'train', 'test1').  Otherwise return the full name so different-
    prefix groups (e.g. comparing ptb_full vs wiki2_full) stay readable.
    """
    prefix = group_output_name(group)
    if name.startswith(prefix + '_'):
        return name[len(prefix) + 1:]
    return name


def is_train_split(name):
    """True when the name looks like a training split (ends with _train or _trainN)."""
    suffix = name.rsplit('_', 1)[-1]
    return suffix == 'train' or (suffix.startswith('train') and suffix[5:].isdigit())
