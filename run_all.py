#!/usr/bin/env python

import argparse
import os
import pickle
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

import data


def main():
    tok_choices = sorted(data.TOKENIZER_REGISTRY)

    parser = argparse.ArgumentParser(
        description="Compute complex-system properties of LM datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    req = parser.add_argument_group("required")
    req.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help=(
            "Registry key for the corpus "
            "(e.g. dataset/text/english/wiki2/.full). "
            "Valid keys are listed in src/data/loader.py."
        ),
    )
    req.add_argument(
        "--save_path",
        required=True,
        metavar="NAME",
        help="Stem used for all output filenames (e.g. ptb_full).",
    )

    # ── Tokenization ──────────────────────────────────────────────────────────
    tok = parser.add_argument_group("tokenization")
    tok.add_argument(
        "--tokenizer",
        default="word",
        choices=tok_choices,
        metavar="NAME",
        help=f"Tokenizer to apply. Available: {{{', '.join(tok_choices)}}}.",
    )

    # ── Analysis selection ────────────────────────────────────────────────────
    sel = parser.add_argument_group("analysis selection")
    sel.add_argument(
        "--processes",
        nargs="+",
        default=["mi", "heaps", "taylors", "ebelings", "recurrence"],
        choices=["mi", "pmi", "heaps", "taylors", "ebelings", "recurrence"],
        metavar="ANALYSIS",
        help=(
            "Analyses to run (space-separated). "
            "Choices: mi pmi heaps taylors ebelings recurrence. "
            "heaps/taylors/ebelings are computed by the same C binary; "
            "selecting any of them runs it. "
            "pmi is expensive and omitted from the default."
        ),
    )

    # ── MI / PMI parameters ───────────────────────────────────────────────────
    mi_grp = parser.add_argument_group("MI / PMI parameters")
    mi_grp.add_argument(
        "--cutoff",
        type=int,
        default=1000,
        metavar="D",
        help="Maximum distance D for MI / PMI computation.",
    )
    mi_grp.add_argument(
        "--mi_method",
        default="grassberger",
        choices=["grassberger", "standard"],
        help="Entropy estimator used for MI.",
    )
    mi_grp.add_argument(
        "--log_base",
        default="2",
        choices=["e", "2", "10"],
        help="Logarithm base for MI / PMI values.",
    )
    mi_grp.add_argument(
        "--threads",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel worker processes (MI) / threads (PMI, recurrence).",
    )
    mi_grp.add_argument(
        "--no_overlap",
        action="store_true",
        help="Disallow overlapping token windows (default: overlaps are allowed).",
    )
    mi_grp.add_argument(
        "--clear",
        action="store_true",
        help="Delete any existing LDD file before running MI (forces a fresh run).",
    )

    # ── Output paths ──────────────────────────────────────────────────────────
    out = parser.add_argument_group("output paths")
    out.add_argument(
        "--experiments_dir",
        default=None,
        metavar="DIR",
        help=(
            "Root directory for all outputs. "
            "Defaults to experiments/<data_type>/<language>/<tokenizer>/."
        ),
    )
    out.add_argument(
        "--data_type",
        default=None,
        metavar="TYPE",
        help="Override the data-type path component (auto-detected from registry).",
    )
    out.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="Override the language path component (auto-detected from registry).",
    )
    out.add_argument(
        "--pmi_dir",
        default=None,
        metavar="DIR",
        help="Output directory for PMI matrices (overrides the default inside experiments_dir).",
    )

    args = parser.parse_args()

    # ── Validate dataset path ─────────────────────────────────────────────────
    from data.loader import _REGISTRY

    if args.data not in _REGISTRY:
        parser.error(
            f"Unknown data path '{args.data}'." " Valid paths are listed in src/data/loader.py."
        )

    # ── Resolve tokenizer and derived settings ────────────────────────────────
    tokenizer = data.get_tokenizer(args.tokenizer)
    log_type = {"e": 0, "2": 1, "10": 2}[args.log_base]
    overlap = 0 if args.no_overlap else 1

    # ── Resolve output directory ──────────────────────────────────────────────
    if args.experiments_dir:
        exp_dir = args.experiments_dir
    else:
        reg_label, reg_data_type, reg_language, *_ = _REGISTRY[args.data]
        data_type = args.data_type or reg_data_type
        language = args.language or reg_language
        exp_dir = os.path.join("experiments", data_type, language, tokenizer.name)

    for subdir in [
        "datasetInIDs",
        "corpus",
        "zipf",
        "ldds",
        "heaps",
        "taylors",
        "ebelings",
        "recurrence",
        "pmi",
    ]:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    ###########################################################################
    # Load corpus (always required — feeds every analysis)
    ###########################################################################

    corpus = data.Corpus(args.data, tokenizer)
    np.savetxt(
        os.path.join(exp_dir, "datasetInIDs", args.save_path + ".csv"),
        np.asarray(corpus.sequentialData.dataArray, dtype=np.uint32),
        fmt="%d",
    )

    corpus_path = os.path.join(exp_dir, "corpus", args.save_path + "_corpus.dat")
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)
    with open(corpus_path, "rb") as f:
        _ = pickle.load(f)

    print("Corpus loaded")

    ###########################################################################
    # Zipf's Law (always computed — negligible cost)
    ###########################################################################

    ranked = corpus.dictionary.counter.most_common()
    IDs, frequency = zip(*ranked) if ranked else ([], [])
    np.savez_compressed(
        os.path.join(exp_dir, "zipf", args.save_path),
        ids=np.asarray(IDs),
        frequency=np.asarray(frequency),
    )
    print("Zipf computed")

    ###########################################################################
    # Mutual Information / LDD
    ###########################################################################

    if "mi" in args.processes:
        from analysis import mi

        ldd_path = os.path.join(exp_dir, "ldds", args.save_path + ".csv")
        if args.clear:
            try:
                os.remove(ldd_path)
            except OSError:
                print(f"Note: {ldd_path} did not exist, nothing to clear.")
        mi.MutualInformation(
            corpus,
            log_type,
            args.threads,
            ldd_path,
            overlap,
            args.mi_method,
            args.cutoff,
        )
        print("LDDs computed")

    ###########################################################################
    # Pointwise Mutual Information
    ###########################################################################

    if "pmi" in args.processes:
        from analysis import pmi as pmi_mod

        pmi_dir = args.pmi_dir or os.path.join(exp_dir, "pmi", args.save_path)
        os.makedirs(pmi_dir, exist_ok=True)
        pmi_mod.PointwiseMutualInformation(
            corpus,
            log_type,
            args.threads,
            pmi_dir,
            overlap,
            "standard",
            args.cutoff,
        )
        print("PMI computed")

    ###########################################################################
    # Heap's Law, Taylor's Law, Ebeling's Method  (C binary — runs all three)
    ###########################################################################

    _scaling = {"heaps", "taylors", "ebelings"}
    if _scaling & set(args.processes):
        binary = "src/analysis/scaling_laws"
        if not os.path.exists(binary):
            print("Compiling src/analysis/scaling_laws.c ...")
            result = subprocess.run(
                ["gcc", "-O2", "-o", binary, "src/analysis/scaling_laws.c", "-lm"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print("Compilation failed:\n" + result.stderr)
            else:
                print("Compiled src/analysis/scaling_laws.c")
        if os.path.exists(binary):
            subprocess.run([binary, args.save_path, exp_dir])
            print("Heaps / Taylor's / Ebeling's computed")

    ###########################################################################
    # Recurrence
    ###########################################################################

    if "recurrence" in args.processes:
        from analysis import recurrence as rec_mod

        recur = rec_mod.Recurrence(corpus, args.threads)
        rec_path = os.path.join(exp_dir, "recurrence", args.save_path + ".npz")
        np.savez_compressed(rec_path, **{str(k): v for k, v in recur.recurrenceList.items()})
        print("Recurrence computed")


if __name__ == "__main__":
    main()
