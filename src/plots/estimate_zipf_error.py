"""
Fit a power law to ordered Zipf arrays and report MSE / MAE for each file.

Reads <experiments_dir>/zipf/<name>.npz (named keys 'ids'/'frequency', with
fallback to 'arr_0'/'arr_1' for older files).

Usage:
    python estimate_zipf_error.py --save_paths ptb_train ptb_ordered_test ptb_ordered_valid
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from plot_utils import powerlaw, fit_powerlaw, SEABORN_STYLE


def load_zipf(path):
    with np.load(path) as arr:
        if 'ids' in arr:
            return arr['ids'], arr['frequency']
        return arr['arr_0'], arr['arr_1']


def main():
    parser = argparse.ArgumentParser(description='Estimate power law fit error for Zipf distributions')
    parser.add_argument('--experiments_dir', default='experiments',
                        help='Experiments root directory (default: experiments)')
    parser.add_argument('--save_paths', nargs='+', required=True,
                        help='List of save-path names to evaluate')
    parser.add_argument('--show', action='store_true',
                        help='Show interactive plot for each file')
    args = parser.parse_args()

    zipf_dir = os.path.join(args.experiments_dir, 'zipf')

    for name in args.save_paths:
        path = os.path.join(zipf_dir, name + '.npz')
        if not os.path.exists(path):
            print(f'Skipping {name}: file not found ({path})')
            continue

        ids, frequency = load_zipf(path)
        xdata = np.arange(1, len(ids) + 1, dtype=float)
        ydata = frequency.astype(float)

        amp, index, covar = fit_powerlaw(xdata, ydata)
        index_err = np.sqrt(covar[1][1]) if covar is not None else float('nan')
        amp_err = np.sqrt(covar[0][0]) * amp if covar is not None else float('nan')

        fit_ydata = powerlaw(xdata, amp, index)
        mse = mean_squared_error(ydata, fit_ydata)
        mae = mean_absolute_error(ydata, fit_ydata)

        print(f'{name}: index={index:.4f} ± {index_err:.4f}, '
              f'amp={amp:.4f} ± {amp_err:.4f}, MSE={mse:.4f}, MAE={mae:.4f}')

        if args.show:
            with plt.style.context(SEABORN_STYLE):
                plt.loglog(xdata, ydata, label='data')
                plt.loglog(xdata, fit_ydata, label=f'fit α={index:.3f}')
                plt.title(name)
                plt.xlabel('Rank')
                plt.ylabel('Frequency')
                plt.legend()
                plt.show()
                plt.clf()


if __name__ == '__main__':
    main()
