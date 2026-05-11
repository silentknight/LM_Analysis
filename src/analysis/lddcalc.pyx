import numpy as np
cimport numpy as np
import scipy.sparse


cpdef getJointRV(dataArray, unsigned long[:] lineLengthList, int totalLength, int d, int overlap):

	cdef unsigned long index = 0
	cdef unsigned long nLines = lineLengthList.shape[0]
	cdef unsigned long i = 0
	cdef unsigned long line_len = 0

	# Zero-copy view of input buffer — avoids the 2×N copy from array.extend
	data_np = np.frombuffer(dataArray, dtype=np.uint64)

	if overlap == 1:
		if totalLength // d <= 1:
			return None, None, None, None, None
		X_arr = data_np[:totalLength - d]
		Y_arr = data_np[d:totalLength]
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
			return None, None, None, None, None
		X_arr = np.concatenate(x_parts)
		Y_arr = np.concatenate(y_parts)

	if len(X_arr) == 0:
		return None, None, None, None, None

	unique_X, counts_X = np.unique(X_arr, return_counts=True)
	unique_Y, counts_Y = np.unique(Y_arr, return_counts=True)

	cdef unsigned long max_x = int(unique_X[-1]) + 1
	cdef unsigned long max_y = int(unique_Y[-1]) + 1

	# Reinterpret uint64 as int64 without copying (word IDs always fit in int64).
	# view() is safe here because both types are 8 bytes and slices are contiguous.
	X_i64 = X_arr.view(np.int64) if X_arr.flags['C_CONTIGUOUS'] else X_arr.astype(np.int64)
	Y_i64 = Y_arr.view(np.int64) if Y_arr.flags['C_CONTIGUOUS'] else Y_arr.astype(np.int64)

	sort_idx = np.lexsort((Y_i64, X_i64))
	X_sorted = X_i64[sort_idx]
	Y_sorted = Y_i64[sort_idx]
	del sort_idx  # free index array before allocating diff_mask

	diff_mask = np.empty(len(X_sorted), dtype=np.bool_)
	diff_mask[0] = True
	diff_mask[1:] = (X_sorted[1:] != X_sorted[:-1]) | (Y_sorted[1:] != Y_sorted[:-1])

	pair_rows = X_sorted[diff_mask]
	pair_cols = Y_sorted[diff_mask]
	del X_sorted, Y_sorted  # free sorted arrays before building sparse matrix

	change_pos = np.where(diff_mask)[0]
	n_total = len(X_arr)
	del diff_mask

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
