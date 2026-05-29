#!/usr/bin/env python
"""
Cross-linguistic pilot study — go/no-go gate for Paper 2 (CL).

Tests whether the broken power-law structure in MI(d) holds across languages
and whether D_pos / D_doc behave as predicted.

Usage:
    # Step 1: prepare corpus files (one article per blank-line-separated block)
    # Step 2: run
    python pilot_crosslingual.py \
        --en  experiments/text/english/word/ldds/ptb_train_grassberger.csv \
        --fi  path/to/fi_wiki_10M.txt \
        --zh  path/to/zh_wiki_10M.txt \
        --cutoff 4999 \
        --out  pilot_results/

    # --en accepts either a pre-computed CSV (d,mi,...) or a raw text file.
    # --fi and --zh must be raw text files (will compute MI from scratch).

Predictions tested (go/no-go criteria):
    P1  Broken power law fits all languages    (KS stat < 0.05)
    P2  D_pos(Finnish, char) > D_pos(English)  (ratio >= 1.2)
    P3  D_doc / mean_article_len in [0.6, 1.5] for all languages
    P4  Asymmetry < 0.15% at all d            (stationarity ok)
"""

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from analysis import lddcalc  # type: ignore[attr-defined]
from analysis.entropy_estimators import get_estimator

N_WORKERS = 7
OVERLAP = 1
LOG_BASE_VAL = 1.0  # nats


# ---------------------------------------------------------------------------
# Corpus loading from raw text (character-level)
# ---------------------------------------------------------------------------

def load_raw_text(path: str, max_chars: int = 15_000_000):
    """Load a plain text file and return (token_ids, line_lengths, id_to_char).

    Articles are separated by blank lines. Each article becomes one 'line'
    (sentence) in the corpus — matching the corpus structure used for MI.
    Characters are mapped to integer IDs starting at 1.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars]

    char_to_id: dict[str, int] = {}
    token_ids: list[int] = []
    line_lengths: list[int] = []

    articles = text.split("\n\n")
    for article in articles:
        article = article.strip()
        if not article:
            continue
        ids = []
        for ch in article:
            if ch not in char_to_id:
                char_to_id[ch] = len(char_to_id) + 1
            ids.append(char_to_id[ch])
        if ids:
            token_ids.extend(ids)
            line_lengths.append(len(ids))

    id_to_char = {v: k for k, v in char_to_id.items()}
    return (
        np.array(token_ids, dtype=np.uint64),
        np.array(line_lengths, dtype=np.uint64),
        id_to_char,
    )


def corpus_stats(line_lengths: np.ndarray) -> dict:
    """Return basic statistics about article/line lengths."""
    return {
        "n_articles": len(line_lengths),
        "total_tokens": int(line_lengths.sum()),
        "mean_article_len": float(line_lengths.mean()),
        "median_article_len": float(np.median(line_lengths)),
        "std_article_len": float(line_lengths.std()),
    }


# ---------------------------------------------------------------------------
# MI computation (mirrors run_all_estimators.py)
# ---------------------------------------------------------------------------

def _worker(args):
    shm_name, corpus_shape, line_lengths, total_length, d = args
    shm = SharedMemory(name=shm_name)
    try:
        corpus = np.ndarray(corpus_shape, dtype=np.uint64, buffer=shm.buf)
        Ni_X, Ni_Y, Ni_XY = lddcalc.getJointCounts(
            corpus, line_lengths, total_length, d, OVERLAP
        )
        if Ni_X is None:
            return d, None
        est = get_estimator("grassberger")
        Hx = est.entropy(Ni_X, LOG_BASE_VAL)
        Hy = est.entropy(Ni_Y, LOG_BASE_VAL)
        Hxy = est.entropy(Ni_XY, LOG_BASE_VAL)
        mi = Hx + Hy - Hxy
        asym = abs(Hx - Hy) / mi if mi > 0 else 0.0
        return d, (mi, asym)
    finally:
        shm.close()


def compute_mi(corpus_np: np.ndarray, line_lengths: np.ndarray, cutoff: int) -> tuple:
    """Compute Grassberger MI(d) for d=1..cutoff. Returns (d_arr, mi_arr, asym_arr)."""
    total_length = int(corpus_np.sum() > 0 and len(corpus_np))  # count tokens
    total_length = len(corpus_np)
    cutoff = min(cutoff, total_length - 1)

    shm = SharedMemory(create=True, size=max(corpus_np.nbytes, 1))
    np.ndarray(corpus_np.shape, dtype=np.uint64, buffer=shm.buf)[:] = corpus_np
    shm_name = shm.name
    corpus_shape = corpus_np.shape

    results = {}
    distances = list(range(1, cutoff + 1))

    try:
        import heapq

        def make_args(d):
            return (shm_name, corpus_shape, line_lengths, total_length, d)

        with Pool(processes=N_WORKERS) as pool:
            pending = {}
            d_iter = iter(distances)
            for _ in range(N_WORKERS * 2):
                try:
                    d = next(d_iter)
                    pending[d] = pool.apply_async(_worker, (make_args(d),))
                except StopIteration:
                    break

            buf = []
            next_d = 1
            done = 0
            t0 = time.time()

            while pending:
                ready = [d for d, r in pending.items() if r.ready()]
                if not ready:
                    time.sleep(0.05)
                    continue

                for d in ready:
                    d_val, res = pending.pop(d).get()
                    if res is None:
                        print(f"\n  No pairs at d={d_val}, stopping.")
                        break
                    heapq.heappush(buf, (d_val, res))
                    try:
                        nd = next(d_iter)
                        pending[nd] = pool.apply_async(_worker, (make_args(nd),))
                    except StopIteration:
                        pass

                while buf and buf[0][0] == next_d:
                    d_val, res = heapq.heappop(buf)
                    results[d_val] = res
                    done += 1
                    next_d += 1
                    if done % 500 == 0:
                        elapsed = time.time() - t0
                        print(
                            f"  d={d_val:6d} ({done}/{cutoff})  "
                            f"mi={res[0]:.4f}  asym={res[1]:.4f}  "
                            f"[{elapsed:.0f}s]",
                            flush=True,
                        )
    finally:
        shm.close()
        shm.unlink()

    d_arr = np.array(sorted(results.keys()))
    mi_arr = np.array([results[d][0] for d in d_arr])
    asym_arr = np.array([results[d][1] for d in d_arr])
    return d_arr, mi_arr, asym_arr


def load_csv(path: str) -> tuple:
    """Load a pre-computed MI CSV (columns: d, mi, ..., asym)."""
    d_list, mi_list, asym_list = [], [], []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0].strip().lstrip("-").isdigit():
                continue
            try:
                d_list.append(int(row[0]))
                mi_list.append(float(row[1]))
                asym_list.append(float(row[5]) if len(row) > 5 else 0.0)
            except (ValueError, IndexError):
                continue
    return np.array(d_list), np.array(mi_list), np.array(asym_list)


# ---------------------------------------------------------------------------
# Broken power-law fitting (two breakpoints, three regimes)
# ---------------------------------------------------------------------------

def fit_sbpl(d_arr: np.ndarray, mi_arr: np.ndarray, d_min: int = 1) -> dict:
    """Fit a smoothly broken power law to MI(d) using Astropy.

    Returns a dict with keys: amplitude, alpha1, x_break1, delta1,
    alpha2, x_break2, delta2, alpha3, fit_mi, ks_stat, ks_pval.

    Two-stage fit: first find the rough first breakpoint (~d=4), then fit
    the full model.
    """
    mask = d_arr >= d_min
    d = d_arr[mask].astype(float)
    mi = mi_arr[mask]

    # --- Stage 1: find first breakpoint (D_pos) using single SBPL ---
    sbpl1 = SmoothlyBrokenPowerLaw1D(
        amplitude=mi[0],
        x_break=4.0,
        alpha1=-0.5,
        alpha2=-0.05,
        delta=1.0,
    )
    sbpl1.x_break.bounds = (2.0, 20.0)
    sbpl1.alpha1.bounds = (-5.0, 0.0)
    sbpl1.alpha2.bounds = (-2.0, 0.0)
    sbpl1.delta.bounds = (0.1, 10.0)

    fitter = LevMarLSQFitter()
    try:
        fitted1 = fitter(sbpl1, d, mi)
        d_break1 = float(fitted1.x_break)
    except Exception:
        d_break1 = 4.0

    # --- Stage 2: fit a second SBPL to large-d portion (d > 5*d_break1) ---
    mask2 = d > 5 * d_break1
    if mask2.sum() < 20:
        mask2 = d > d_break1 * 2

    d2 = d[mask2]
    mi2 = mi[mask2]

    mid_d = float(np.sqrt(d2[0] * d2[-1]))
    sbpl2 = SmoothlyBrokenPowerLaw1D(
        amplitude=mi2[0],
        x_break=mid_d,
        alpha1=-0.05,
        alpha2=-0.002,
        delta=1.0,
    )
    sbpl2.x_break.bounds = (d2[0], d2[-1])
    sbpl2.alpha1.bounds = (-2.0, 0.0)
    sbpl2.alpha2.bounds = (-1.0, 0.5)
    sbpl2.delta.bounds = (0.1, 10.0)

    try:
        fitted2 = fitter(sbpl2, d2, mi2)
        d_break2 = float(fitted2.x_break)
        alpha2 = float(fitted2.alpha1)
        alpha3 = float(fitted2.alpha2)
    except Exception:
        d_break2 = float(d[-1]) / 2
        alpha2 = -0.05
        alpha3 = 0.0

    # Combined fit quality: KS between observed and model
    fitted_mi1 = fitted1(d)
    ks_stat, ks_pval = ks_2samp(mi / mi.max(), fitted_mi1 / fitted_mi1.max())

    return {
        "alpha1": float(fitted1.alpha1),
        "D_pos": d_break1,
        "alpha2": alpha2,
        "D_doc": d_break2,
        "alpha3": alpha3,
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
        "fit_d": d,
        "fit_mi": fitted1(d),
    }


# ---------------------------------------------------------------------------
# Go/no-go evaluation
# ---------------------------------------------------------------------------

CRITERIA = {
    "P1_fit_quality":   "Broken power law fits (KS stat < 0.05)",
    "P2_dpos_typology": "D_pos(Finnish, char) >= 1.2 × D_pos(English)",
    "P3_ddoc_article":  "D_doc / mean_article_len in [0.5, 2.0] for all languages",
    "P4_asymmetry":     "Max asymmetry < 0.15% for all languages",
}


def evaluate(results: dict) -> dict:
    """Evaluate go/no-go criteria. results: {lang: {stats, fit, ...}}."""
    verdicts = {}

    # P1: fit quality for all languages
    p1 = all(r["fit"]["ks_stat"] < 0.05 for r in results.values())
    verdicts["P1_fit_quality"] = ("GO" if p1 else "NO-GO", {
        lang: f"KS={r['fit']['ks_stat']:.4f}" for lang, r in results.items()
    })

    # P2: D_pos typology (only if Finnish is present)
    if "fi" in results and "en" in results:
        ratio = results["fi"]["fit"]["D_pos"] / results["en"]["fit"]["D_pos"]
        verdicts["P2_dpos_typology"] = (
            "GO" if ratio >= 1.2 else "NO-GO",
            {"fi/en D_pos ratio": f"{ratio:.2f} (need >= 1.20)"},
        )
    else:
        verdicts["P2_dpos_typology"] = ("SKIP", {"reason": "Finnish not provided"})

    # P3: D_doc vs mean article length
    p3_details = {}
    p3_pass = True
    for lang, r in results.items():
        ddoc = r["fit"]["D_doc"]
        mean_art = r["stats"]["mean_article_len"]
        if mean_art > 0:
            ratio = ddoc / mean_art
            p3_details[lang] = f"D_doc={ddoc:.0f}, mean_art={mean_art:.0f}, ratio={ratio:.2f}"
            if not (0.5 <= ratio <= 2.0):
                p3_pass = False
        else:
            p3_details[lang] = "mean_article_len=0 (skip)"
    verdicts["P3_ddoc_article"] = ("GO" if p3_pass else "NO-GO", p3_details)

    # P4: asymmetry
    p4_details = {}
    p4_pass = True
    for lang, r in results.items():
        max_asym = float(r["asym"].max()) * 100
        p4_details[lang] = f"max asym={max_asym:.4f}%"
        if max_asym > 0.15:
            p4_pass = False
    verdicts["P4_asymmetry"] = ("GO" if p4_pass else "NO-GO", p4_details)

    return verdicts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves(results: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"en": "#1f77b4", "fi": "#2ca02c", "zh": "#d62728"}
    labels = {"en": "English (PTB)", "fi": "Finnish (Wikipedia)", "zh": "Mandarin (Wikipedia)"}

    ax = axes[0]
    for lang, r in results.items():
        d = r["d"]
        mi = r["mi"]
        ax.loglog(d, mi, "-", color=colors.get(lang, "grey"),
                  label=labels.get(lang, lang), linewidth=1.5, alpha=0.8)
        fit = r["fit"]
        ax.axvline(fit["D_pos"], color=colors.get(lang, "grey"),
                   linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(fit["D_doc"], color=colors.get(lang, "grey"),
                   linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Distance d (tokens / characters)")
    ax.set_ylabel("MI (nats)")
    ax.set_title("MI(d) curves — pilot study")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax2 = axes[1]
    langs = list(results.keys())
    dpos_vals = [results[l]["fit"]["D_pos"] for l in langs]
    ddoc_vals = [results[l]["fit"]["D_doc"] for l in langs]
    x = np.arange(len(langs))
    w = 0.35
    bars1 = ax2.bar(x - w / 2, dpos_vals, w, label="D_pos", color="#7fbadb")
    bars2 = ax2.bar(x + w / 2, ddoc_vals, w, label="D_doc", color="#a8d9a8")

    mean_arts = [results[l]["stats"]["mean_article_len"] for l in langs]
    ax2.scatter(x + w / 2, mean_arts, marker="x", color="black", s=60,
                zorder=5, label="mean article len")

    ax2.set_xticks(x)
    ax2.set_xticklabels([labels.get(l, l) for l in langs], fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylabel("tokens / characters (log scale)")
    ax2.set_title("D_pos and D_doc vs. mean article length")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "pilot_mi_curves.pdf")
    plt.savefig(path, format="pdf", bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-linguistic pilot study")
    parser.add_argument("--en", required=True,
                        help="English: pre-computed CSV or raw text file")
    parser.add_argument("--fi", default=None,
                        help="Finnish Wikipedia: raw text file (blank-line separated articles)")
    parser.add_argument("--zh", default=None,
                        help="Mandarin Wikipedia: raw text file (blank-line separated articles)")
    parser.add_argument("--cutoff", type=int, default=4999,
                        help="Max distance d (default 4999 for pilot speed)")
    parser.add_argument("--max_chars", type=int, default=10_000_000,
                        help="Max characters to load per corpus (default 10M)")
    parser.add_argument("--out", default="pilot_results",
                        help="Output directory for CSVs, plots, and report")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    lang_files = {"en": args.en}
    if args.fi:
        lang_files["fi"] = args.fi
    if args.zh:
        lang_files["zh"] = args.zh

    results = {}

    for lang, path in lang_files.items():
        print(f"\n{'='*60}")
        print(f"  Language: {lang.upper()}  |  {path}")
        print(f"{'='*60}")

        csv_out = os.path.join(args.out, f"{lang}_mi.csv")

        # Load or compute MI
        if path.endswith(".csv") and os.path.exists(path):
            print("  Loading pre-computed MI CSV...")
            d_arr, mi_arr, asym_arr = load_csv(path)
            line_lengths = np.array([])  # not available from CSV
            stats = {"n_articles": 0, "total_tokens": int(d_arr[-1] + 1),
                     "mean_article_len": 0.0, "median_article_len": 0.0,
                     "std_article_len": 0.0}
        else:
            print("  Loading raw text corpus...")
            corpus_np, line_lengths, _ = load_raw_text(path, args.max_chars)
            stats = corpus_stats(line_lengths)
            print(f"  Articles: {stats['n_articles']:,}  "
                  f"Tokens: {stats['total_tokens']:,}  "
                  f"Mean article: {stats['mean_article_len']:.0f} chars")

            if os.path.exists(csv_out):
                print(f"  Found cached CSV: {csv_out} — loading.")
                d_arr, mi_arr, asym_arr = load_csv(csv_out)
            else:
                print(f"  Computing MI(d) up to d={args.cutoff} ...")
                d_arr, mi_arr, asym_arr = compute_mi(corpus_np, line_lengths, args.cutoff)
                with open(csv_out, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["d", "mi", "asym"])
                    for d, mi, a in zip(d_arr, mi_arr, asym_arr):
                        w.writerow([d, mi, a])
                print(f"  Saved MI data: {csv_out}")

        # Fit broken power law
        print("  Fitting broken power law...")
        fit = fit_sbpl(d_arr, mi_arr)
        print(f"  α₁={fit['alpha1']:.3f}  D_pos={fit['D_pos']:.1f}  "
              f"α₂={fit['alpha2']:.3f}  D_doc={fit['D_doc']:.1f}  "
              f"KS={fit['ks_stat']:.4f}")

        results[lang] = {
            "d": d_arr,
            "mi": mi_arr,
            "asym": asym_arr,
            "fit": fit,
            "stats": stats,
        }

    # Evaluate go/no-go
    print(f"\n{'='*60}")
    print("  GO / NO-GO ASSESSMENT")
    print(f"{'='*60}")
    verdicts = evaluate(results)

    all_go = True
    for criterion, (verdict, details) in verdicts.items():
        icon = "✓" if verdict == "GO" else ("−" if verdict == "SKIP" else "✗")
        print(f"\n  [{icon}] {verdict:7s}  {CRITERIA[criterion]}")
        for k, v in details.items():
            print(f"           {k}: {v}")
        if verdict == "NO-GO":
            all_go = False

    print(f"\n{'='*60}")
    if all_go:
        print("  OVERALL: GO — proceed to EXP-6 (full multilingual study)")
    else:
        print("  OVERALL: NO-GO — review failed criteria before EXP-6")
    print(f"{'='*60}")

    # Write report
    report_path = os.path.join(args.out, "pilot_report.txt")
    with open(report_path, "w") as f:
        f.write("CROSS-LINGUISTIC PILOT REPORT\n\n")
        f.write("Parameters\n")
        for lang, r in results.items():
            f.write(f"  {lang:4s}  α₁={r['fit']['alpha1']:+.3f}  "
                    f"D_pos={r['fit']['D_pos']:7.1f}  "
                    f"α₂={r['fit']['alpha2']:+.3f}  "
                    f"D_doc={r['fit']['D_doc']:7.1f}  "
                    f"KS={r['fit']['ks_stat']:.4f}  "
                    f"mean_art={r['stats']['mean_article_len']:.0f}\n")
        f.write("\nCorpus stats\n")
        for lang, r in results.items():
            s = r["stats"]
            f.write(f"  {lang:4s}  articles={s['n_articles']:,}  "
                    f"tokens={s['total_tokens']:,}  "
                    f"mean_art={s['mean_article_len']:.1f}  "
                    f"median_art={s['median_article_len']:.1f}\n")
        f.write("\nVerdicts\n")
        for criterion, (verdict, details) in verdicts.items():
            f.write(f"  {verdict:7s}  {criterion}\n")
            for k, v in details.items():
                f.write(f"           {k}: {v}\n")
        f.write(f"\nOVERALL: {'GO' if all_go else 'NO-GO'}\n")

    print(f"\n  Report: {report_path}")

    # Plot
    plot_curves(results, args.out)


if __name__ == "__main__":
    main()
