import numpy as np
cimport numpy as np
import scipy.sparse


def _extract_pairs(dataArray, unsigned long[:] lineLengthList, int totalLength, int d, int overlap):
    """Shared pair-extraction logic for both MI and PMI paths."""
    cdef unsigned long index = 0
    cdef unsigned long nLines = lineLengthList.shape[0]
    cdef unsigned long i = 0
    cdef unsigned long line_len = 0

    # Accept both array.array (PMI path) and numpy array (MI shared-memory path)
    if isinstance(dataArray, np.ndarray):
        data_np = dataArray if dataArray.dtype == np.uint64 else dataArray.view(np.uint64)
    else:
        data_np = np.frombuffer(dataArray, dtype=np.uint64)

    if overlap == 1:
        if totalLength // d <= 1:
            return None, None
        return data_np[:totalLength - d], data_np[d:totalLength]
    else:
        x_parts = []
        y_parts = []
        for i in range(nLines):
            line_len = lineLengthList[i]
            if line_len // d > 1:
                x_parts.append(data_np[index:index + line_len - d])
                y_parts.append(data_np[index + d:index + line_len])
            index += line_len
        if not x_parts:
            return None, None
        return np.concatenate(x_parts), np.concatenate(y_parts)


cpdef getJointCounts(dataArray, unsigned long[:] lineLengthList, int totalLength, int d, int overlap):
    """
    MI path — returns (Ni_X, Ni_Y, pair_counts) without building a sparse matrix.

    Encodes each (X, Y) pair as a single uint64 and sorts in-place, keeping peak
    memory to ~1×N×8 bytes instead of the 3×N×8 bytes needed by the old lexsort.
    """
    X_arr, Y_arr = _extract_pairs(dataArray, lineLengthList, totalLength, d, overlap)
    if X_arr is None or len(X_arr) == 0:
        return None, None, None

    unique_X, counts_X = np.unique(X_arr, return_counts=True)
    unique_Y, counts_Y = np.unique(Y_arr, return_counts=True)

    # Encode: X*V_enc + Y uniquely identifies each pair when Y < V_enc.
    # Safe for V_enc < 2^32 (all practical NLP vocabularies satisfy this).
    cdef unsigned long V_enc = int(unique_Y[-1]) + 1
    encoded = X_arr * np.uint64(V_enc) + Y_arr  # one N×8-byte allocation
    encoded.sort()                                # in-place, no extra N-element array

    diff_mask = np.empty(len(encoded), dtype=np.bool_)
    diff_mask[0] = True
    diff_mask[1:] = encoded[1:] != encoded[:-1]

    change_pos = np.where(diff_mask)[0]
    n_total = len(encoded)
    del diff_mask, encoded  # free N×8 + N bytes before building result

    pair_counts = np.empty(len(change_pos), dtype=np.int64)
    if len(change_pos) > 1:
        pair_counts[:-1] = np.diff(change_pos)
    if len(change_pos) > 0:
        pair_counts[-1] = n_total - change_pos[-1]
    del change_pos

    return counts_X.astype(np.float64), counts_Y.astype(np.float64), pair_counts.astype(np.float64)


cpdef getJointRV(dataArray, unsigned long[:] lineLengthList, int totalLength, int d, int overlap):
    """
    PMI path — returns (Ni_X, Ni_Y, XY_sparse, unique_X, unique_Y).

    Uses the same encoded in-place sort as getJointCounts; recovers pair (row, col)
    from the unique encoded values via integer division/modulo.
    """
    X_arr, Y_arr = _extract_pairs(dataArray, lineLengthList, totalLength, d, overlap)
    if X_arr is None or len(X_arr) == 0:
        return None, None, None, None, None

    unique_X, counts_X = np.unique(X_arr, return_counts=True)
    unique_Y, counts_Y = np.unique(Y_arr, return_counts=True)

    cdef unsigned long max_x = int(unique_X[-1]) + 1
    cdef unsigned long max_y = int(unique_Y[-1]) + 1
    cdef unsigned long V_enc = max_y  # Y < V_enc guarantees injectivity

    encoded = X_arr * np.uint64(V_enc) + Y_arr
    encoded.sort()  # in-place

    diff_mask = np.empty(len(encoded), dtype=np.bool_)
    diff_mask[0] = True
    diff_mask[1:] = encoded[1:] != encoded[:-1]

    unique_enc = encoded[diff_mask]             # K unique pairs
    change_pos = np.where(diff_mask)[0]
    n_total = len(encoded)
    del diff_mask, encoded

    # Recover X and Y from the encoded value
    pair_rows = (unique_enc // np.uint64(V_enc)).astype(np.int64)
    pair_cols = (unique_enc %  np.uint64(V_enc)).astype(np.int64)
    del unique_enc

    pair_counts = np.empty(len(change_pos), dtype=np.int64)
    if len(change_pos) > 1:
        pair_counts[:-1] = np.diff(change_pos)
    if len(change_pos) > 0:
        pair_counts[-1] = n_total - change_pos[-1]
    del change_pos

    XY = scipy.sparse.csc_matrix(
        (pair_counts.astype(np.float64), (pair_rows, pair_cols)),
        shape=(max_x, max_y))

    return counts_X.astype(np.float64), counts_Y.astype(np.float64), XY, unique_X, unique_Y


cpdef getStandardPMI(P_XY_data, P_XY_row, P_XY_col, PX, PY,
                     unsigned long dataLen, unsigned long pxLen, unsigned long pyLen, int base):

    cdef extern from "math.h":
        cdef double log(double x)
        cdef double log2(double x)
        cdef double log10(double x)

    cdef double[:] data = P_XY_data
    cdef unsigned long[:] row = P_XY_row
    cdef unsigned long[:] col = P_XY_col
    cdef double[:] px = PX
    cdef double[:] py = PY
    cdef unsigned long i = 0
    cdef double denominator

    pmi = np.zeros(dataLen, dtype=np.float64)
    cdef double[:] temp_pmi = pmi

    for i in range(dataLen):
        denominator = px[row[i]] * py[col[i]]
        if base == 0:
            temp_pmi[i] = log(data[i] / denominator)
        elif base == 1:
            temp_pmi[i] = log2(data[i] / denominator)
        elif base == 2:
            temp_pmi[i] = log10(data[i] / denominator)

    return pmi
