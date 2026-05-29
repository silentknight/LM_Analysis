#!/usr/bin/env python

import csv
import os
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from tqdm import tqdm

from . import lddcalc  # type: ignore[attr-defined]
from .entropy_estimators import get_estimator


def _mi_worker(
    shm_name,
    corpus_shape,
    line_length_list,
    total_length,
    d,
    overlap,
    log_type,
    method,
    q,
):
    """
    Worker process: attaches to the shared-memory corpus (zero copy), computes
    MI(d) via getJointCounts, and puts (d, mi, Hx, Hy, Hxy) into the queue.
    Runs in a separate process so all CPU cores are used without GIL contention.
    """
    shm = SharedMemory(name=shm_name)
    try:
        corpus = np.ndarray(corpus_shape, dtype=np.uint64, buffer=shm.buf)
        Ni_X, Ni_Y, Ni_XY = lddcalc.getJointCounts(
            corpus, line_length_list, total_length, d, overlap
        )

        if Ni_X is None:
            q.put((d, None, None, None, None))
            return

        # log_base_val: divisor to convert nats → target unit (1.0 for nats, ln2 for bits, ln10 for hartleys)
        log_base_val = (1.0, np.log(2), np.log(10))[log_type]

        estimator = get_estimator(method)
        Hx = estimator.entropy(Ni_X, log_base_val)
        Hy = estimator.entropy(Ni_Y, log_base_val)
        Hxy = estimator.entropy(Ni_XY, log_base_val)
        mi_val = Hx + Hy - Hxy

        q.put((d, float(mi_val), float(Hx), float(Hy), float(Hxy)))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"MI worker failed at d={d}: {exc}")
        q.put((d, None, None, None, None))
    finally:
        shm.close()


class MutualInformation(object):
    def __init__(
        self,
        corpusData,
        log_type,
        no_of_threads,
        data_file_path,
        overlap,
        method,
        cutoff,
    ):
        self.corpus = corpusData
        self.line_length_list = np.array(corpusData.sequentialData.wordCountList, dtype=np.uint64)
        self.total_length = corpusData.sequentialData.totalLength
        self.no_of_threads = no_of_threads
        self.filename = data_file_path
        self.overlap = overlap
        self.method = method
        self.log_type = log_type
        self.cutoff = cutoff

        get_estimator(method)  # validates method name before spawning any workers

        # Copy corpus into shared memory once; worker processes attach zero-copy.
        corpus_np = np.array(corpusData.sequentialData.dataArray, dtype=np.uint64)
        self.corpus_shape = corpus_np.shape
        self.shm = SharedMemory(create=True, size=max(corpus_np.nbytes, 1))
        np.ndarray(corpus_np.shape, dtype=np.uint64, buffer=self.shm.buf)[:] = corpus_np
        del corpus_np

        try:
            self.mutualInformation = self.calculate_MI()
        finally:
            self.shm.close()
            self.shm.unlink()

    def calculate_MI(self):
        mi, Hx, Hy, Hxy = [], [], [], []
        d = 1

        print("Average String Length: ", int(self.corpus.sequentialData.averageLength))
        print("Total String Length:   ", int(self.corpus.sequentialData.totalLength))

        # Resume from previously saved file if it exists
        if os.path.exists(self.filename):
            with open(self.filename, "r", newline="") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                    if (
                        len(header) >= 2
                        and header[0] == "data"
                        and header[1] == self.corpus.datainfo
                    ):
                        next(reader)  # skip column-header row
                        for row in reader:
                            if len(row) < 5:
                                print("Warning: skipping malformed row: " + str(row))
                                continue
                            try:
                                row_d = int(row[0])
                                while len(mi) < row_d - 1:
                                    mi.append(float("nan"))
                                    Hx.append(float("nan"))
                                    Hy.append(float("nan"))
                                    Hxy.append(float("nan"))
                                mi.append(float(row[1]))
                                Hx.append(float(row[2]))
                                Hy.append(float(row[3]))
                                Hxy.append(float(row[4]))
                                d = row_d + 1
                            except (IndexError, ValueError):
                                print("Warning: skipping malformed row: " + str(row))
                except StopIteration:
                    pass
        else:
            print("File does not exist to load previous data")

        # Atomically rewrite clean file (avoids truncation on crash)
        tmp_path = self.filename + ".tmp"
        with open(tmp_path, "w", newline="") as f_tmp:
            w = csv.writer(f_tmp)
            w.writerow(["data", self.corpus.datainfo])
            w.writerow(["d", "mi", "Hx", "Hy", "Hxy"])
            for i in range(len(mi)):
                if not np.isnan(mi[i]):
                    w.writerow([i + 1, mi[i], Hx[i], Hy[i], Hxy[i]])
        os.replace(tmp_path, self.filename)

        max_distance = self.total_length
        total_d = min(self.cutoff, max_distance - 1)
        if d > 1:
            print(f"Resuming MI from d={d}")

        end = False
        procs = []
        q = Queue()

        with open(self.filename, "a", newline="") as f:
            try:
                with tqdm(
                    total=total_d,
                    initial=d - 1,
                    desc="MI (LDD)",
                    unit="d",
                    dynamic_ncols=True,
                ) as pbar:
                    while d < max_distance and d <= self.cutoff and not end:

                        mi.extend([0.0] * self.no_of_threads)
                        Hx.extend([0.0] * self.no_of_threads)
                        Hy.extend([0.0] * self.no_of_threads)
                        Hxy.extend([0.0] * self.no_of_threads)

                        # Launch one process per distance in this batch
                        procs = [
                            Process(
                                target=_mi_worker,
                                args=(
                                    self.shm.name,
                                    self.corpus_shape,
                                    self.line_length_list,
                                    self.total_length,
                                    d + i,
                                    self.overlap,
                                    self.log_type,
                                    self.method,
                                    q,
                                ),
                                daemon=True,
                            )
                            for i in range(self.no_of_threads)
                        ]
                        for p in procs:
                            p.start()

                        # Collect results — workers may finish out of order
                        results = {}
                        for _ in range(self.no_of_threads):
                            try:
                                res = q.get(timeout=600)
                            except Exception:
                                end = True
                                for p in procs:
                                    if p.is_alive():
                                        p.terminate()
                                break
                            results[res[0]] = res[1:]  # keyed by distance

                        for p in procs:
                            p.join(timeout=5)
                            if p.is_alive():
                                p.kill()
                                p.join()

                        # Process results in ascending distance order
                        for i in range(self.no_of_threads):
                            if d + i > self.cutoff:
                                threads_remaining = self.no_of_threads - i
                                if threads_remaining > 0:
                                    del mi[-threads_remaining:]
                                    del Hx[-threads_remaining:]
                                    del Hy[-threads_remaining:]
                                    del Hxy[-threads_remaining:]
                                end = True
                                break
                            res = results.get(d + i, (None, None, None, None))
                            if res[0] is None or np.isnan(res[0]):
                                end = True
                                threads_remaining = self.no_of_threads - i
                                if threads_remaining > 0:
                                    del mi[-threads_remaining:]
                                    del Hx[-threads_remaining:]
                                    del Hy[-threads_remaining:]
                                    del Hxy[-threads_remaining:]
                                break

                            mi[d + i - 1] = res[0]
                            Hx[d + i - 1] = res[1]
                            Hy[d + i - 1] = res[2]
                            Hxy[d + i - 1] = res[3]

                            csv.writer(f).writerow(
                                [
                                    d + i,
                                    mi[d + i - 1],
                                    Hx[d + i - 1],
                                    Hy[d + i - 1],
                                    Hxy[d + i - 1],
                                ]
                            )
                            f.flush()
                            pbar.set_postfix(d=d + i, mi=f"{res[0]:.4f}")
                            pbar.update(1)

                        d += self.no_of_threads

            except KeyboardInterrupt:
                print(f"\nInterrupted at d={d - 1}")
                for p in procs:
                    p.terminate()

        return np.array(mi), np.array(Hx), np.array(Hy), np.array(Hxy)
