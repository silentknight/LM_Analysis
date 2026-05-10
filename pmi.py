#!/usr/bin/env python

import numpy as np
import scipy.special as spec
import sys
import threading
import os
import speedup as lddCalc
import array
import scipy.sparse as sp


class myThread(threading.Thread):
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
		super(myThread, self).__init__()

	def run(self):
		Ni_X, Ni_Y, Ni_XY, self.Xi, self.Yi = lddCalc.getJointRV(
			self.data_array, self.line_length_list, self.total_length, self.d, self.overlap)

		if Ni_X is None:
			self.complete = True
			return

		if self.method == "standard":
			P_XY = Ni_XY / np.sum(Ni_XY)
			P_X = (Ni_X / np.sum(Ni_X)).toarray()[0]
			P_Y = (Ni_Y / np.sum(Ni_Y)).toarray()[0]
			P_XY = P_XY.tocoo()
			pmi_data = lddCalc.getStandardPMI(
				P_XY.data, np.uint64(P_XY.row), np.uint64(P_XY.col),
				P_X, P_Y,
				np.uint64(P_XY.data.size), np.uint64(P_X.size), np.uint64(P_Y.size),
				self.log_type)
			self.pmi = sp.coo_matrix((pmi_data, (P_XY.row, P_XY.col)), shape=P_XY.shape).tocsc()
			self.Ni_X = Ni_X
			self.Ni_Y = Ni_Y
			self.Ni_XY = Ni_XY


class PointwiseMutualInformation(object):
	def __init__(self, corpusData, log_type, no_of_threads, data_file_path, overlap, method, cutoff):
		self.corpus = corpusData
		self.data_array = array.array('L', corpusData.sequentialData.dataArray)
		self.line_length_list = np.array(corpusData.sequentialData.wordCountList, dtype=np.uint64)
		self.total_length = corpusData.sequentialData.totalLength
		self.no_of_threads = no_of_threads
		self.directory = data_file_path
		self.overlap = overlap
		self.method = method
		self.log_type = log_type
		self.cutoff = cutoff
		self.pointwiseMutualInformation = self.calculate_PMI()

	def calculate_PMI(self):
		d = 1
		print("Average String Length: ", int(self.corpus.sequentialData.averageLength))
		print("Total String Length: ", int(self.corpus.sequentialData.totalLength))

		# Resume from last saved distance if the output directory already has results
		try:
			files = sorted(os.listdir(os.path.join(self.directory, "np")))
			d_nums = [int(f.split('.')[0]) for f in files if f.endswith('.npz')]
			if d_nums:
				d = max(d_nums) + 1
		except (FileNotFoundError, OSError, ValueError, IndexError):
			print(self.directory + " does not exist or cannot be loaded; starting fresh.")

		with open(os.path.join(self.directory, "0.symbols.dat"), "w") as f:
			f.write(str(self.corpus.dictionary.word2idx))

		os.makedirs(os.path.join(self.directory, "np"), exist_ok=True)
		os.makedirs(os.path.join(self.directory, "Ni_XY"), exist_ok=True)
		os.makedirs(os.path.join(self.directory, "pmi"), exist_ok=True)

		end = False
		distances_computed = 0

		try:
			max_distance = self.total_length
			while d < max_distance and d <= self.cutoff and not end:

				thread = []
				for i in range(self.no_of_threads):
					thread.append(myThread(
						d + i, self.overlap, self.log_type, self.method,
						self.data_array, self.line_length_list, self.total_length))

				for i in range(self.no_of_threads):
					thread[i].start()

				# Join all threads before reading results to avoid data races
				for i in range(self.no_of_threads):
					thread[i].join()

				for i in range(self.no_of_threads):
					if thread[i].complete:
						end = True
						break
					np.savez(os.path.join(self.directory, "np", str(d + i)),
					         thread[i].Xi, thread[i].Yi, thread[i].Ni_X, thread[i].Ni_Y)
					sp.save_npz(os.path.join(self.directory, "Ni_XY", str(d + i)), thread[i].Ni_XY)
					sp.save_npz(os.path.join(self.directory, "pmi", str(d + i)), thread[i].pmi)
					distances_computed += 1

				d += self.no_of_threads

		except KeyboardInterrupt:
			print("Processed upto: " + str(d))

		return distances_computed
