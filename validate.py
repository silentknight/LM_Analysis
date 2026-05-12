#!/usr/bin/env python
"""
Baseline validation tool.

Save current experiments as a named baseline:
    python validate.py save --save_path ptb_full \
        --run_cmd "python -u run_all.py --data dataset/ptb/.full --words 1 ..."

Check (re-runs stored command then compares against baseline):
    python validate.py check --save_path ptb_full

Check without re-running (compare existing outputs only):
    python validate.py check --save_path ptb_full --skip_run
"""

import argparse
import os
import sys
import csv
import shutil
import subprocess

import numpy as np

BASELINE_DIR = "baseline"

# Tolerances per output type.
# rel_tol: max(|new-ref|/|ref|) allowed
# abs_tol: max(|new-ref|) allowed (used when ref ≈ 0)
TOLERANCES = {
    "datasetInIDs": dict(exact=True),
    "zipf": dict(exact=True),
    "ldds": dict(abs_tol=1e-10),
    "heaps": dict(exact=True),
    "taylors": dict(abs_tol=1e-6, rel_tol=0.01),
    "ebelings": dict(abs_tol=1e-4, rel_tol=0.01),
    "recurrence": dict(structural=True),
}


def _bdir(save_path):
    return os.path.join(BASELINE_DIR, save_path)


def _bpath(save_path, subdir, filename):
    return os.path.join(_bdir(save_path), subdir, filename)


def _result(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    line = f"  [{tag}] {name}"
    if detail:
        line += f": {detail}"
    print(line)
    return ok


# ---------------------------------------------------------------------------
# comparators
# ---------------------------------------------------------------------------


def cmp_dataset_ids(new_path, ref_path):
    def load(p):
        with open(p) as f:
            return [x for x in f.read().split() if x.strip()]

    new = load(new_path)
    ref = load(ref_path)
    if new == ref:
        return True, f"{len(new):,} tokens match"
    short, long_ = (new, ref) if len(new) <= len(ref) else (ref, new)
    if short == long_[: len(short)] and len(long_) - len(short) <= 1:
        return (
            True,
            f"{len(short):,} tokens match (extra trailing: {len(long_)-len(short)})",
        )
    diffs = sum(a != b for a, b in zip(new, ref))
    return False, f"{diffs:,} differences, lengths {len(new):,} vs {len(ref):,}"


def cmp_zipf(new_path, ref_path):
    a = np.load(new_path)
    b = np.load(ref_path)

    def get(npz, named, positional):
        return npz[named] if named in npz else npz[positional]

    a_ids = get(a, "ids", "arr_0")
    a_freq = get(a, "frequency", "arr_1")
    b_ids = get(b, "ids", "arr_0")
    b_freq = get(b, "frequency", "arr_1")
    ids_ok = np.array_equal(a_ids, b_ids)
    freq_ok = np.array_equal(a_freq, b_freq)
    ok = ids_ok and freq_ok
    parts = []
    if not ids_ok:
        parts.append("ids differ")
    if not freq_ok:
        parts.append("frequency differs")
    return ok, "exact match" if ok else "; ".join(parts)


def cmp_ldd_csv(new_path, ref_path, abs_tol=1e-10, **_):
    def load(p):
        with open(p, newline="") as f:
            r = csv.reader(f)
            next(r)
            next(r)
            return {
                int(row[0]): [float(x) for x in row[1:5]] for row in r if len(row) >= 5
            }

    new = load(new_path)
    ref = load(ref_path)
    if set(new) != set(ref):
        return False, f"distance sets differ: {len(new)} vs {len(ref)}"
    max_abs = max(abs(nv - rv) for d in new for nv, rv in zip(new[d], ref[d]))
    ok = max_abs <= abs_tol
    return ok, f"max_abs={max_abs:.2e}  tol={abs_tol:.0e}"


def cmp_heaps(new_path, ref_path, **_):
    def load(p):
        with open(p) as f:
            return [x for x in f.read().split(",") if x.strip()]

    new = load(new_path)
    ref = load(ref_path)
    short, long_ = (new, ref) if len(new) <= len(ref) else (ref, new)
    if short == long_[: len(short)] and len(long_) - len(short) <= 1:
        return (
            True,
            f"{len(short):,} values match (extra trailing: {len(long_)-len(short)})",
        )
    diffs = sum(a != b for a, b in zip(new, ref))
    return False, f"{diffs:,} value differences, lengths {len(new):,} vs {len(ref):,}"


def cmp_taylors(new_path, ref_path, abs_tol=1e-6, rel_tol=0.01, **_):
    def load(p):
        rows = []
        with open(p) as f:
            for line in f:
                line = line.strip().rstrip(",")
                if not line:
                    continue
                try:
                    rows.append(("n", int(line)))
                except ValueError:
                    rows.append(("data", np.array([float(x) for x in line.split(",")])))
        return rows

    new_rows = load(new_path)
    ref_rows = load(ref_path)
    if len(new_rows) != len(ref_rows):
        return False, f"row count {len(new_rows)} vs {len(ref_rows)}"
    max_abs = max_rel = 0.0
    for nr, rr in zip(new_rows, ref_rows):
        if nr[0] != rr[0]:
            return False, "row type mismatch"
        if nr[0] == "n":
            if nr[1] != rr[1]:
                return False, f"window size mismatch: {nr[1]} vs {rr[1]}"
        else:
            diff = np.abs(nr[1] - rr[1])
            max_abs = max(max_abs, float(diff.max()))
            nz = np.abs(rr[1]) > 1e-30
            if nz.any():
                max_rel = max(max_rel, float((diff[nz] / np.abs(rr[1][nz])).max()))
    ok = max_abs <= abs_tol or max_rel <= rel_tol
    return (
        ok,
        f"max_abs={max_abs:.2e}  max_rel={max_rel:.2e}"
        f"  tol=abs:{abs_tol:.0e}/rel:{rel_tol:.0%}",
    )


def cmp_ebelings(new_path, ref_path, abs_tol=1e-4, rel_tol=0.01, **_):
    def load(p):
        with open(p) as f:
            return {
                int(t.split(":")[0]): float(t.split(":")[1])
                for t in f.read().split(",")
                if ":" in t
            }

    new = load(new_path)
    ref = load(ref_path)
    if set(new) != set(ref):
        return False, "key sets differ"
    max_abs = max_rel = 0.0
    for k in new:
        d = abs(new[k] - ref[k])
        max_abs = max(max_abs, d)
        if abs(ref[k]) > 1e-30:
            max_rel = max(max_rel, d / abs(ref[k]))
    ok = max_abs <= abs_tol or max_rel <= rel_tol
    return (
        ok,
        f"max_abs={max_abs:.2e}  max_rel={max_rel:.2e}"
        f"  tol=abs:{abs_tol:.0e}/rel:{rel_tol:.0%}",
    )


def cmp_recurrence(new_path, ref_path, **_):
    def total_tokens(p):
        r = np.load(p, allow_pickle=True)
        total = 0
        for k in r.keys():
            v = r[k]
            total += int(np.sum(v[1])) + 1 if v.size > 0 else 1
        return total

    nt = total_tokens(new_path)
    rt = total_tokens(ref_path)
    return nt == rt, f"tokens: new={nt:,}  ref={rt:,}"


# ---------------------------------------------------------------------------
# file manifest  (name, experiments_src, baseline_subdir, comparator)
# ---------------------------------------------------------------------------


def manifest(sp):
    return [
        (
            "datasetInIDs",
            f"experiments/datasetInIDs/{sp}.csv",
            "datasetInIDs",
            f"{sp}.csv",
            cmp_dataset_ids,
        ),
        ("zipf", f"experiments/zipf/{sp}.npz", "zipf", f"{sp}.npz", cmp_zipf),
        ("ldds", f"experiments/ldds/{sp}.csv", "ldds", f"{sp}.csv", cmp_ldd_csv),
        (
            "heaps",
            f"experiments/heaps/{sp}_heaps.csv",
            "heaps",
            f"{sp}_heaps.csv",
            cmp_heaps,
        ),
        (
            "taylors",
            f"experiments/taylors/{sp}_taylors.csv",
            "taylors",
            f"{sp}_taylors.csv",
            cmp_taylors,
        ),
        (
            "ebelings",
            f"experiments/ebelings/{sp}_ebelings.csv",
            "ebelings",
            f"{sp}_ebelings.csv",
            cmp_ebelings,
        ),
        (
            "recurrence",
            f"experiments/recurrence/{sp}.npz",
            "recurrence",
            f"{sp}.npz",
            cmp_recurrence,
        ),
    ]


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


def cmd_save(args):
    sp = args.save_path
    print(f"Saving baseline for '{sp}' → {_bdir(sp)}/")
    saved = 0
    for name, src, subdir, fname, _ in manifest(sp):
        dst = _bpath(sp, subdir, fname)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  saved  {name}")
            saved += 1
        else:
            print(f"  skip   {name}  (not found: {src})")

    if args.run_cmd:
        cmd_file = os.path.join(_bdir(sp), "cmd.txt")
        os.makedirs(_bdir(sp), exist_ok=True)
        with open(cmd_file, "w") as f:
            f.write(args.run_cmd.strip() + "\n")
        print("  saved  run command → cmd.txt")

    total = len(manifest(sp))
    print(f"Done. {saved}/{total} files saved.")


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def cmd_check(args):
    sp = args.save_path
    bdir = _bdir(sp)

    if not os.path.isdir(bdir):
        print(f"No baseline found at '{bdir}'. Run 'save' first.")
        sys.exit(1)

    if not args.skip_run:
        cmd_file = os.path.join(bdir, "cmd.txt")
        if not os.path.exists(cmd_file):
            print("No cmd.txt in baseline. Use --skip_run or re-save with --run_cmd.")
            sys.exit(1)
        with open(cmd_file) as f:
            run_cmd = f.read().strip()
        print(f"Re-running: {run_cmd}\n")
        ret = subprocess.run(run_cmd, shell=True)
        if ret.returncode != 0:
            print("Run command failed — aborting check.")
            sys.exit(1)
        print()

    tol = TOLERANCES
    print(f"Comparing against baseline '{bdir}/'")
    all_pass = True
    for name, src, subdir, fname, cmp_fn in manifest(sp):
        ref_path = _bpath(sp, subdir, fname)
        if not os.path.exists(ref_path):
            print(f"  [SKIP] {name}: not in baseline")
            continue
        if not os.path.exists(src):
            _result(name, False, "output file missing")
            all_pass = False
            continue
        try:
            ok, detail = cmp_fn(
                src,
                ref_path,
                **{
                    k: v
                    for k, v in tol.get(name, {}).items()
                    if k not in ("exact", "structural")
                },
            )
        except Exception as e:
            ok, detail = False, f"error: {e}"
        if not ok:
            all_pass = False
        _result(name, ok, detail)

    print()
    print("RESULT:", "ALL PASS" if all_pass else "FAILURES DETECTED")
    sys.exit(0 if all_pass else 1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Baseline validation for LM_Analysis outputs"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_save = sub.add_parser(
        "save", help="Save current experiments/ outputs as baseline"
    )
    p_save.add_argument("--save_path", required=True)
    p_save.add_argument(
        "--run_cmd",
        default="",
        help="Command string to store and replay during 'check'",
    )

    p_check = sub.add_parser(
        "check", help="Re-run analysis and compare against baseline"
    )
    p_check.add_argument("--save_path", required=True)
    p_check.add_argument(
        "--skip_run",
        action="store_true",
        help="Skip re-running; compare existing experiments/ against baseline",
    )

    args = parser.parse_args()
    {"save": cmd_save, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    main()
