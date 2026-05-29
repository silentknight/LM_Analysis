#!/usr/bin/env python

import array
import os
import threading

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from . import lddcalc  # type: ignore[attr-defined]
from .entropy_estimators import ESTIMATOR_NAMES, get_estimator

# These estimators correct global entropy bias but do not alter per-symbol probability
# estimates, so PMI values are identical to "standard" for them.
_PMI_MLE_EQUIVALENT = {"standard", "miller_madow", "grassberger", "grassberger_2003", "nsb"}


class MyThread(threading.Thread):
    def __init__(self, d, overlap, log_type, method, data_array, line_length_list, total_length):
        self.d = d
        self.Ni_X = 0
        self.Ni_Y = 0
        self.Ni_XY = 0
        self.pmi = 0
        self.Xi = None
        self.Yi = None
        self.complete = False
        self.overlap = overlap
        self.method = method
        self.log_type = log_type
        self.data_array = data_array
        self.line_length_list = line_length_list
        self.total_length = total_length
        super(MyThread, self).__init__()

    def run(self):
        try:
            self._run()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"PMI thread failed at d={self.d}: {exc}")
            self.complete = True  # prevents the outer loop from treating this as a valid result

    def _run(self):
        Ni_X, Ni_Y, Ni_XY, self.Xi, self.Yi = lddcalc.getJointRV(
            self.data_array,
            self.line_length_list,
            self.total_length,
            self.d,
            self.overlap,
        )

        if Ni_X is None:
            self.complete = True
            return

        estimator = get_estimator(self.method)

        # getStandardPMI indexes P_X/P_Y by raw vocab ID, so build vocab-length arrays.
        Xi = self.Xi.astype(np.int64)
        Yi = self.Yi.astype(np.int64)
        P_X = np.zeros(int(Xi[-1]) + 1, dtype=np.float64)
        P_X[Xi] = estimator.probabilities(Ni_X)
        P_Y = np.zeros(int(Yi[-1]) + 1, dtype=np.float64)
        P_Y[Yi] = estimator.probabilities(Ni_Y)

        P_XY_coo = Ni_XY.tocoo()
        p_xy_data = estimator.probabilities(P_XY_coo.data)

        pmi_data = lddcalc.getStandardPMI(
            p_xy_data,
            np.uint64(P_XY_coo.row),
            np.uint64(P_XY_coo.col),
            P_X,
            P_Y,
            np.uint64(p_xy_data.size),
            np.uint64(P_X.size),
            np.uint64(P_Y.size),
            self.log_type,
        )
        self.pmi = sp.coo_matrix((pmi_data, (P_XY_coo.row, P_XY_coo.col)), shape=P_XY_coo.shape).tocsc()
        self.Ni_X = Ni_X
        self.Ni_Y = Ni_Y
        self.Ni_XY = Ni_XY


class PointwiseMutualInformation(object):
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
        self.data_array = array.array("Q", corpusData.sequentialData.dataArray)
        self.line_length_list = np.array(corpusData.sequentialData.wordCountList, dtype=np.uint64)
        self.total_length = corpusData.sequentialData.totalLength
        self.no_of_threads = no_of_threads
        self.directory = data_file_path
        self.overlap = overlap
        self.method = method
        self.log_type = log_type
        self.cutoff = cutoff
        if method not in ESTIMATOR_NAMES:
            raise ValueError(f"Unknown PMI method: {method!r}. Choose from: {ESTIMATOR_NAMES}")
        if method in _PMI_MLE_EQUIVALENT and method != "standard":
            print(
                f"Note: --mi_method {method!r} corrects entropy bias but not per-symbol "
                "probabilities. PMI values will be identical to 'standard'."
            )
        self.pointwiseMutualInformation = self.calculate_PMI()

    def calculate_PMI(self):
        d = 1
        print("Average String Length: ", int(self.corpus.sequentialData.averageLength))
        print("Total String Length: ", int(self.corpus.sequentialData.totalLength))

        os.makedirs(os.path.join(self.directory, "marginals"), exist_ok=True)
        os.makedirs(os.path.join(self.directory, "Ni_XY"), exist_ok=True)
        os.makedirs(os.path.join(self.directory, "pmi"), exist_ok=True)

        # Resume from last saved distance if the output directory already has results
        try:
            files = sorted(os.listdir(os.path.join(self.directory, "marginals")))
            d_nums = [int(f.split(".")[0]) for f in files if f.endswith(".npz")]
            if d_nums:
                d = max(d_nums) + 1
        except (FileNotFoundError, OSError, ValueError, IndexError):
            print(self.directory + " does not exist or cannot be loaded; starting fresh.")

        with open(os.path.join(self.directory, "0.symbols.csv"), "w") as f:
            f.write("word,id\n")
            for word, idx in self.corpus.dictionary.word2idx.items():
                f.write(f"{word},{idx}\n")

        end = False
        distances_computed = 0
        max_distance = self.total_length
        total_d = min(self.cutoff, max_distance - 1)
        if d > 1:
            print(f"Resuming PMI from d={d}")

        try:
            with tqdm(
                total=total_d, initial=d - 1, desc="PMI", unit="d", dynamic_ncols=True
            ) as pbar:
                while d < max_distance and d <= self.cutoff and not end:

                    thread = []
                    for i in range(self.no_of_threads):
                        thread.append(
                            MyThread(
                                d + i,
                                self.overlap,
                                self.log_type,
                                self.method,
                                self.data_array,
                                self.line_length_list,
                                self.total_length,
                            )
                        )

                    for i in range(self.no_of_threads):
                        thread[i].start()

                    # Join all threads before reading results to avoid data races
                    for i in range(self.no_of_threads):
                        thread[i].join()

                    for i in range(self.no_of_threads):
                        if d + i > self.cutoff:
                            end = True
                            break
                        if thread[i].complete:
                            end = True
                            break
                        np.savez_compressed(
                            os.path.join(self.directory, "marginals", str(d + i)),
                            Xi=thread[i].Xi,
                            Yi=thread[i].Yi,
                            Ni_X=thread[i].Ni_X,
                            Ni_Y=thread[i].Ni_Y,
                        )
                        sp.save_npz(
                            os.path.join(self.directory, "Ni_XY", str(d + i)),
                            thread[i].Ni_XY,
                        )
                        sp.save_npz(
                            os.path.join(self.directory, "pmi", str(d + i)),
                            thread[i].pmi,
                        )
                        distances_computed += 1
                        pbar.update(1)

                    d += self.no_of_threads

        except KeyboardInterrupt:
            print(f"\nInterrupted at d={d}")

        return distances_computed
