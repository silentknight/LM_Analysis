#!/usr/bin/env python
"""
Single-pass multi-estimator LDD script (train split only).
Extracts joint counts once per d, applies all estimators, writes in-order.
Output columns: d, mi, Hx, Hy, Hxy, asym  (asym = |Hx-Hy|/MI quality flag)
"""

import csv
import heapq
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from analysis import lddcalc  # type: ignore[attr-defined]
from analysis.entropy_estimators import ESTIMATOR_NAMES, get_estimator

DATA_KEY = "dataset/text/english/ptb/.train"
CUTOFF = 9999
N_WORKERS = 7
OVERLAP = 1
LOG_BASE_VAL = 1.0  # nats (matching --log_base e)
OUT_DIR = "experiments/text/english/word/ldds"

ESTIMATORS = [n for n in ESTIMATOR_NAMES if n != "nsb"]


def _worker(args):
    shm_name, corpus_shape, line_lengths, total_length, d = args
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(name=shm_name)
    try:
        corpus = np.ndarray(corpus_shape, dtype=np.uint64, buffer=shm.buf)
        Ni_X, Ni_Y, Ni_XY = lddcalc.getJointCounts(
            corpus, line_lengths, total_length, d, OVERLAP
        )
        if Ni_X is None:
            return d, None
        results = {}
        for name in ESTIMATORS:
            est = get_estimator(name)
            Hx = est.entropy(Ni_X, LOG_BASE_VAL)
            Hy = est.entropy(Ni_Y, LOG_BASE_VAL)
            Hxy = est.entropy(Ni_XY, LOG_BASE_VAL)
            mi = Hx + Hy - Hxy
            asym = abs(Hx - Hy) / mi if mi > 0 else 0.0
            results[name] = (mi, Hx, Hy, Hxy, asym)
        return d, results
    finally:
        shm.close()


def main():
    from multiprocessing.shared_memory import SharedMemory

    import data
    from data.tokenizers import get_tokenizer

    print("Loading corpus...")
    tokenizer = get_tokenizer("word")
    corpus = data.Corpus(DATA_KEY, tokenizer)
    total_length = corpus.sequentialData.totalLength
    line_lengths = np.array(corpus.sequentialData.wordCountList, dtype=np.uint64)
    print(f"Total tokens: {total_length}")

    corpus_np = np.array(corpus.sequentialData.dataArray, dtype=np.uint64)
    shm = SharedMemory(create=True, size=max(corpus_np.nbytes, 1))
    np.ndarray(corpus_np.shape, dtype=np.uint64, buffer=shm.buf)[:] = corpus_np
    shm_name = shm.name
    corpus_shape = corpus_np.shape
    del corpus_np

    os.makedirs(OUT_DIR, exist_ok=True)

    writers = {}
    files = {}
    for name in ESTIMATORS:
        path = os.path.join(OUT_DIR, f"ptb_train_{name}.csv")
        f = open(path, "w", newline="")
        w = csv.writer(f)
        w.writerow(["data", DATA_KEY])
        w.writerow(["d", "mi", "Hx", "Hy", "Hxy", "asym"])
        writers[name] = w
        files[name] = f

    cutoff = min(CUTOFF, total_length - 1)
    distances = list(range(1, cutoff + 1))

    def make_args(d):
        return (shm_name, corpus_shape, line_lengths, total_length, d)

    buf = []  # min-heap of (d, results_dict)
    next_d = 1
    done = 0
    sentinel_file = files[ESTIMATORS[0]]

    try:
        with Pool(processes=N_WORKERS) as pool:
            pending = {}
            d_iter = iter(distances)

            for _ in range(N_WORKERS * 2):
                try:
                    d = next(d_iter)
                    pending[d] = pool.apply_async(_worker, (make_args(d),))
                except StopIteration:
                    break

            while pending:
                ready = [d for d, r in pending.items() if r.ready()]
                if not ready:
                    time.sleep(0.05)
                    continue

                for d in ready:
                    d_val, results = pending.pop(d).get()
                    if results is None:
                        print(f"\nNo pairs at d={d_val}, stopping.")
                        return
                    heapq.heappush(buf, (d_val, results))

                    try:
                        nd = next(d_iter)
                        pending[nd] = pool.apply_async(_worker, (make_args(nd),))
                    except StopIteration:
                        pass

                # Flush all consecutive ready d-values in order
                while buf and buf[0][0] == next_d:
                    d_val, results = heapq.heappop(buf)
                    for name in ESTIMATORS:
                        mi, Hx, Hy, Hxy, asym = results[name]
                        writers[name].writerow([d_val, mi, Hx, Hy, Hxy, asym])
                    sentinel_file.flush()
                    done += 1
                    next_d += 1
                    if done % 100 == 0:
                        g = results["grassberger"]
                        cs = results["chao_shen"]
                        print(
                            f"  d={d_val:6d}  ({done}/{cutoff})"
                            f"  mi_grassberger={g[0]:.4f}  asym={g[4]:.4f}"
                            f"  mi_chao_shen={cs[0]:.4f}  asym={cs[4]:.4f}",
                            flush=True,
                        )

        # Drain any remaining buffered results (shouldn't happen, but be safe)
        while buf:
            d_val, results = heapq.heappop(buf)
            if d_val != next_d:
                break
            for name in ESTIMATORS:
                mi, Hx, Hy, Hxy, asym = results[name]
                writers[name].writerow([d_val, mi, Hx, Hy, Hxy, asym])
            done += 1
            next_d += 1

    finally:
        for f in files.values():
            f.close()
        shm.close()
        shm.unlink()

    print(f"\nDone. Wrote {done} distances for {len(ESTIMATORS)} estimators.")


if __name__ == "__main__":
    main()
