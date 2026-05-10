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

	# Pre-count XY pairs via integer pair-encoding + np.unique, avoiding an O(N)
	# array of ones and letting numpy handle deduplication before building the matrix
	X_arr = np.asarray(X, dtype=np.int64)
	Y_arr = np.asarray(Y, dtype=np.int64)
	pair_ids = X_arr * max_y + Y_arr
	unique_pairs, pair_counts = np.unique(pair_ids, return_counts=True)
	pair_rows = (unique_pairs // max_y).astype(np.int32)
	pair_cols = (unique_pairs % max_y).astype(np.int32)
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


cpdef getTaylorsLaw(X_data, long l, long start, long end, long vocabularySize):

	cdef unsigned int[:] data = X_data[start:end]
	cdef long no_of_subsequences = X_data[start:end].shape[0]

	results = None
	for i in range(no_of_subsequences):
		[unique, counts] = np.unique(data[i:i + l], return_counts=True)
		entry = scipy.sparse.coo_matrix(
			(counts, (np.ones(len(unique)) * i, unique)),
			shape=(no_of_subsequences, vocabularySize + 1), dtype=np.int32).tocsc()
		if results is None:
			results = entry
		else:
			results += entry

	return results
