import csv
import pickle
import numpy as np


def load_ldd(path):
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


def load_zipf(path):
    with np.load(path) as arr:
        if "ids" in arr:
            return arr["ids"], arr["frequency"]
        return arr["arr_0"], arr["arr_1"]


def load_heaps(path):
    with open(path, "r") as f:
        content = f.read()
    return np.array([int(v) for v in content.split(",") if v.strip()])


def load_ebelings(path):
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


def load_taylors(path, target_subseq_len):
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


def load_corpus(path):
    with open(path, "rb") as f:
        return pickle.load(f)
