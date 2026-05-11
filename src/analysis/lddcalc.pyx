import numpy as np
cimport numpy as np
from cpython cimport array
import array
import scipy.sparse


cpdef getJointRV(dataArray, unsigned long[:] lineLengthList, int totalLength, int d, int overlap):

	cdef array.array X = array.array('L', [])
	cdef array.array Y = array.array('L', [])
	cdef unsigned long index = 0
	cdef unsigned long steps = 0
	cdef unsigned long nLines = lineLengthList.shape[0]
	cdef unsigned long i = 0
	cdef unsigned long line_len = 0

	if overlap == 1:
		# Treat entire dataset as one sequence without mutating the input array
		steps = totalLength // d
		if steps > 1:
			array.extend(X, dataArray[0:totalLength - d])
			array.extend(Y, dataArray[d:totalLength])
	else:
		for i in range(nLines):
			line_len = lineLengthList[i]
			steps = line_len // d
			if steps > 1:
				array.extend(X, dataArray[index:index + line_len - d])
				array.extend(Y, dataArray[index + d:index + line_len])
			index += line_len

	if len(X) == 0 or len(Y) == 0:
		return None, None, None, None, None

	unique_X, counts_X = np.unique(X, return_counts=True)
	unique_Y, counts_Y = np.unique(Y, return_counts=True)

	# unique arrays from np.unique are sorted, so [-1] gives max in O(1) vs max() O(n)
	cdef unsigned long max_x = int(unique_X[-1]) + 1
	cdef unsigned long max_y = int(unique_Y[-1]) + 1

	# Count unique (X, Y) pairs via lexicographic sort — avoids int64 overflow
	# that the previous numeric encoding (X * max_y + Y) could produce for large
	# vocabularies where max_x * max_y > 2^63.
	X_arr = np.asarray(X, dtype=np.int64)
	Y_arr = np.asarray(Y, dtype=np.int64)
	sort_idx = np.lexsort((Y_arr, X_arr))
	X_sorted = X_arr[sort_idx]
	Y_sorted = Y_arr[sort_idx]
	diff_mask = np.empty(len(X_sorted), dtype=np.bool_)
	diff_mask[0] = True
	diff_mask[1:] = (X_sorted[1:] != X_sorted[:-1]) | (Y_sorted[1:] != Y_sorted[:-1])
	pair_rows = X_sorted[diff_mask].astype(np.int64)
	pair_cols = Y_sorted[diff_mask].astype(np.int64)
	boundaries = np.concatenate((np.where(diff_mask)[0], [len(X_sorted)]))
	pair_counts = np.diff(boundaries)
	XY = scipy.sparse.csc_matrix(
		(pair_counts.astype(np.float64), (pair_rows, pair_cols)),
		shape=(max_x, max_y))

	# Return dense float64 arrays for Ni_X / Ni_Y — callers previously wrapped these
	# in 1-row sparse matrices only to extract .data immediately afterwards
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
