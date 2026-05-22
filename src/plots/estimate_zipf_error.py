"""
Fit a power law to ordered Zipf arrays and report MSE / MAE for each file.

Reads <experiments_dir>/zipf/<name>.npz (named keys 'ids'/'frequency', with
fallback to 'arr_0'/'arr_1' for older files).

Usage:
    python estimate_zipf_error.py \
        --save_paths ptb_train ptb_ordered_test ptb_ordered_valid
"""

import os
import sys

_PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PLOTS_DIR)
sys.path.insert(0, os.path.dirname(_PLOTS_DIR))

import argparse

import matplotlib.pyplot as plt
import numpy as np
from plot_utils import SEABORN_STYLE

from utils.loaders import load_zipf
from utils.math_utils import fit_powerlaw, powerlaw


def main():
    parser = argparse.ArgumentParser(
        description="Estimate power law fit error for Zipf distributions"
    )
    parser.add_argument(
        "--experiments_dir",
        required=True,
        help=(
            "Experiments root directory — must point to the leaf tokenizer folder, "
            "e.g. experiments/text/english/word"
        ),
    )
    parser.add_argument(
        "--save_paths",
        nargs="+",
        required=True,
        help="List of save-path names to evaluate",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive plot for each file")
    args = parser.parse_args()

    zipf_dir = os.path.join(args.experiments_dir, "zipf")

    for name in args.save_paths:
        path = os.path.join(zipf_dir, name + ".npz")
        if not os.path.exists(path):
            print(f"Skipping {name}: file not found ({path})")
            continue

        ids, frequency = load_zipf(path)
        xdata = np.arange(1, len(ids) + 1, dtype=float)
        ydata = frequency.astype(float)

        amp, index, covar = fit_powerlaw(xdata, ydata)
        index_err = np.sqrt(covar[1][1]) if covar is not None else float("nan")
        # p[0] = log10(amp), so σ_amp = amp · ln(10) · σ_{p₀} by error propagation
        amp_err = amp * np.log(10) * np.sqrt(covar[0][0]) if covar is not None else float("nan")

        fit_ydata = powerlaw(xdata, amp, index)
        mse = float(np.mean((ydata - fit_ydata) ** 2))
        mae = float(np.mean(np.abs(ydata - fit_ydata)))

        print(
            f"{name}: index={index:.4f} ± {index_err:.4f}, "
            f"amp={amp:.4f} ± {amp_err:.4f}, MSE={mse:.4f}, MAE={mae:.4f}"
        )

        if args.show:
            with plt.style.context(SEABORN_STYLE):
                plt.loglog(xdata, ydata, label="data")
                plt.loglog(xdata, fit_ydata, label=f"fit α={index:.3f}")
                plt.title(name)
                plt.xlabel("Rank")
                plt.ylabel("Frequency")
                plt.legend()
                plt.show()
                plt.clf()


if __name__ == "__main__":
    main()
