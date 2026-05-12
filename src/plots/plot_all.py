#!/usr/bin/env python
"""
Plot complex-system properties for language modelling datasets.

Examples:
    python plot_all.py                          # all plot types
    python plot_all.py --plots zipf ldds        # selected types
    python plot_all.py --plots taylors --subseq_length 500
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt

from datasets import DATASETS
from plot_utils import (
    SEABORN_STYLE,
    powerlaw,
    fit_powerlaw,
    apply_plot_style,
    group_output_name,
    label_for,
    is_train_split,
)

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_ldd(path):
    d_vals, mi_vals = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # data/info header
        next(reader)  # column header (d, mi, Hx, Hy, Hxy)
        for row in reader:
            if row:
                d_vals.append(int(row[0]))
                mi_vals.append(float(row[1]))
    return np.array(d_vals), np.array(mi_vals)


def _load_zipf(path):
    with np.load(path) as arr:
        if "ids" in arr:
            return arr["ids"], arr["frequency"]
        return arr["arr_0"], arr["arr_1"]


def _load_heaps(path):
    with open(path, "r") as f:
        content = f.read()
    return np.array([int(v) for v in content.split(",") if v.strip()])


def _load_ebelings(path):
    subseq_lens, variances = [], []
    with open(path, "r") as f:
        content = f.read()
    for field in content.split(","):
        field = field.strip()
        if not field:
            continue
        parts = field.split(":")
        subseq_lens.append(float(parts[0]))
        variances.append(float(parts[1]))
    return np.array(subseq_lens), np.array(variances)


def _load_taylors(path, target_subseq_len):
    means, sds = [], []
    with open(path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines) - 2:
        try:
            subseq_len = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        if subseq_len == target_subseq_len:
            means = [float(v) for v in lines[i + 1].split(",") if v.strip()]
            sds = [float(v) for v in lines[i + 2].split(",") if v.strip()]
        i += 3
    if not means:
        return None, None
    return np.array(means), np.array(sds)


# ---------------------------------------------------------------------------
# Per-type plot functions  (one group → one saved figure)
# ---------------------------------------------------------------------------


def _plot_ldds(group, exp_dir, out_dir, **_):
    data = {}
    x_lim = 0
    ldd_dir = os.path.join(exp_dir, "ldds")
    for name in group:
        path = os.path.join(ldd_dir, name + ".csv")
        if not os.path.exists(path):
            continue
        d_vals, mi_vals = _load_ldd(path)
        data[name] = (d_vals, mi_vals)
        x_lim = max(x_lim, len(d_vals))
    if not data:
        return

    out_name = group_output_name(group)
    with plt.style.context(SEABORN_STYLE):
        fig, ax = plt.subplots()
        for name, (d_vals, mi_vals) in data.items():
            ax.loglog(d_vals, mi_vals, label=label_for(name, group))
        ax.set_xlim(1, x_lim)
        ax.set_ylim(0.6, 3.5)
        lgd = apply_plot_style(
            ax,
            xlabel="Distance between words, d",
            ylabel="Mutual Information, I(d)",
            legend_loc="upper right",
            legend_ncol=5,
        )
        plt.savefig(
            os.path.join(out_dir, out_name + "_ldds.png"),
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close(fig)


def _plot_zipf(group, exp_dir, out_dir, **_):
    FIT_POINTS = 1000
    zipf_dir = os.path.join(exp_dir, "zipf")
    out_name = group_output_name(group)
    train_index = None

    with plt.style.context(SEABORN_STYLE):
        fig, ax = plt.subplots()
        plotted = False
        for name in group:
            path = os.path.join(zipf_dir, name + ".npz")
            if not os.path.exists(path):
                continue
            ids, frequency = _load_zipf(path)
            xdata = np.arange(1, len(ids) + 1)
            lbl = label_for(name, group)
            n_fit = min(FIT_POINTS, len(xdata))

            if is_train_split(name):
                amp, index, _ = fit_powerlaw(
                    xdata, frequency, pinit=[0, -1.0], n_points=n_fit
                )
                train_index = index
                legend_label = f"{lbl} : α={round(index, 3)}"
            else:
                pinit = [0, train_index] if train_index is not None else [0, -1.0]
                amp, index, _ = fit_powerlaw(
                    xdata, frequency, pinit=pinit, n_points=n_fit
                )
                fit_ydata = powerlaw(xdata.astype(float), amp, index)
                diff = fit_ydata - frequency.astype(float)
                denom = np.std(fit_ydata) * np.std(frequency)
                nmse = float(np.mean(diff**2) / denom) if denom != 0 else float("nan")
                legend_label = f"{lbl} : α={round(index, 3)}, nmse={round(nmse, 3)}"

            ax.loglog(xdata, frequency, label=legend_label)
            plotted = True

        if not plotted:
            plt.close(fig)
            return
        ax.set_xlim(left=1)
        lgd = apply_plot_style(
            ax,
            xlabel="Ordered Rank of Words",
            ylabel="Frequency of Occurrence",
            legend_loc="upper right",
        )
        plt.savefig(
            os.path.join(out_dir, out_name + "_zipf.png"),
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close(fig)


def _plot_heaps(group, exp_dir, out_dir, **_):
    heaps_dir = os.path.join(exp_dir, "heaps")
    out_name = group_output_name(group)

    with plt.style.context(SEABORN_STYLE):
        fig, ax = plt.subplots()
        plotted = False
        for name in group:
            path = os.path.join(heaps_dir, name + "_heaps.csv")
            if not os.path.exists(path):
                continue
            vocab_sizes = _load_heaps(path)
            xdata = np.arange(1, len(vocab_sizes) + 1)
            n_fit = min(len(xdata), max(len(xdata) // 2, 1))
            amp, index, _ = fit_powerlaw(
                xdata, vocab_sizes, pinit=[0, 0.5], n_points=n_fit
            )
            ax.loglog(
                xdata,
                vocab_sizes,
                label=f"{label_for(name, group)} : β={round(index, 3)}",
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=1)
        lgd = apply_plot_style(
            ax,
            xlabel="Length of Subsequence, n",
            ylabel="Size of Vocabulary, V(n)",
            legend_loc="upper left",
        )
        plt.savefig(
            os.path.join(out_dir, out_name + "_heaps.png"),
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close(fig)


def _plot_ebelings(group, exp_dir, out_dir, **_):
    FIT_POINTS = 4
    ebelings_dir = os.path.join(exp_dir, "ebelings")
    out_name = group_output_name(group)

    with plt.style.context(SEABORN_STYLE):
        fig, ax = plt.subplots()
        plotted = False
        for name in group:
            path = os.path.join(ebelings_dir, name + "_ebelings.csv")
            if not os.path.exists(path):
                continue
            subseq_lens, variances = _load_ebelings(path)
            n_fit = min(FIT_POINTS, len(subseq_lens))
            amp, index, _ = fit_powerlaw(
                subseq_lens, variances, pinit=[0, -1.0], n_points=n_fit
            )
            ax.loglog(
                subseq_lens,
                variances,
                label=f"{label_for(name, group)} : η={round(index, 3)}",
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            return
        ax.set_xlim(left=10)
        ax.set_ylim(bottom=10)
        lgd = apply_plot_style(
            ax,
            xlabel="Length of Subsequence",
            ylabel="Variance",
            legend_loc="upper left",
        )
        plt.savefig(
            os.path.join(out_dir, out_name + "_ebelings.png"),
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close(fig)


def _plot_taylors(group, exp_dir, out_dir, subseq_length=1000, **_):
    taylors_dir = os.path.join(exp_dir, "taylors")
    out_name = group_output_name(group)

    with plt.style.context(SEABORN_STYLE):
        fig, ax = plt.subplots()
        plotted = False
        for name in group:
            path = os.path.join(taylors_dir, name + "_taylors.csv")
            if not os.path.exists(path):
                continue
            means, sds = _load_taylors(path, subseq_length)
            if means is None:
                continue
            amp, index, _ = fit_powerlaw(means, sds, pinit=[0, -1.0])
            ax.loglog(
                means,
                powerlaw(means.astype(float), amp, index),
                label=f"{label_for(name, group)} : α={round(index, 3)}",
            )
            ax.scatter(means, sds, s=5)
            plotted = True

        if not plotted:
            plt.close(fig)
            return
        lgd = apply_plot_style(
            ax, xlabel="Mean", ylabel="Standard Deviation", legend_loc="upper left"
        )
        plt.savefig(
            os.path.join(out_dir, out_name + "_taylors.png"),
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

PLOT_TYPES = {
    "ldds": _plot_ldds,
    "zipf": _plot_zipf,
    "heaps": _plot_heaps,
    "ebelings": _plot_ebelings,
    "taylors": _plot_taylors,
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot complex-system properties of LM datasets"
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=list(PLOT_TYPES.keys()),
        choices=list(PLOT_TYPES.keys()),
        help="Plot types to generate (default: all)",
    )
    parser.add_argument(
        "--experiments_dir",
        default="experiments",
        help="Experiments root directory (default: experiments)",
    )
    parser.add_argument(
        "--output_dir", default="plots", help="Directory to save plots (default: plots)"
    )
    parser.add_argument(
        "--subseq_length",
        type=int,
        default=1000,
        help="Subsequence length for Taylor's Law plot (default: 1000)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for plot_type in args.plots:
        fn = PLOT_TYPES[plot_type]
        print(f"--- {plot_type} ---")
        for group in DATASETS:
            out_name = group_output_name(group)
            print(f"  {out_name}")
            fn(
                group,
                exp_dir=args.experiments_dir,
                out_dir=args.output_dir,
                subseq_length=args.subseq_length,
            )


if __name__ == "__main__":
    main()
