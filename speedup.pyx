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

	counts_X = scipy.sparse.coo_matrix(
		(counts_X, (np.zeros(len(unique_X)), unique_X)),
		shape=(1, int(max(unique_X)) + 1), dtype=np.float64).tocsc()
	counts_Y = scipy.sparse.coo_matrix(
		(counts_Y, (np.zeros(len(unique_Y)), unique_Y)),
		shape=(1, int(max(unique_Y)) + 1), dtype=np.float64).tocsc()

	temp_sp = scipy.sparse.coo_matrix(
		(np.ones(len(X)), (np.asarray(X), np.asarray(Y))),
		shape=(max(X) + 1, max(Y) + 1), dtype=np.float64)
	XY = temp_sp.tocsc()

	return counts_X, counts_Y, XY, unique_X, unique_Y


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
