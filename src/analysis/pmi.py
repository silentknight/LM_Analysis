#!/usr/bin/env python

import numpy as np
import threading
import os
import array
import scipy.sparse as sp
from tqdm import tqdm
from . import lddcalc


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
		Ni_X, Ni_Y, Ni_XY, self.Xi, self.Yi = lddcalc.getJointRV(
			self.data_array, self.line_length_list, self.total_length, self.d, self.overlap)

		if Ni_X is None:
			self.complete = True
			return

		if self.method == "standard":
			P_XY = Ni_XY / np.sum(Ni_XY)
			# getStandardPMI indexes px/py using raw vocab IDs (P_XY.row / P_XY.col),
			# so P_X and P_Y must be vocab-ID-indexed arrays of length max_vocab_id+1.
			# self.Xi / self.Yi are unique_X / unique_Y (sorted vocab IDs from getJointRV).
			# Fancy indexing builds the correct layout without constructing sparse matrices.
			P_X = np.zeros(int(self.Xi[-1]) + 1, dtype=np.float64)
			P_X[self.Xi] = Ni_X / np.sum(Ni_X)
			P_Y = np.zeros(int(self.Yi[-1]) + 1, dtype=np.float64)
			P_Y[self.Yi] = Ni_Y / np.sum(Ni_Y)
			P_XY = P_XY.tocoo()
			pmi_data = lddcalc.getStandardPMI(
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

		os.makedirs(os.path.join(self.directory, "marginals"), exist_ok=True)
		os.makedirs(os.path.join(self.directory, "Ni_XY"), exist_ok=True)
		os.makedirs(os.path.join(self.directory, "pmi"), exist_ok=True)

		# Resume from last saved distance if the output directory already has results
		try:
			files = sorted(os.listdir(os.path.join(self.directory, "marginals")))
			d_nums = [int(f.split('.')[0]) for f in files if f.endswith('.npz')]
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
			with tqdm(total=total_d, initial=d - 1, desc="PMI",
			          unit="d", dynamic_ncols=True) as pbar:
				while d < max_distance and d <= self.cutoff and not end:

					thread = []
					for i in range(self.no_of_threads):
						thread.append(MyThread(
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
						np.savez_compressed(os.path.join(self.directory, "marginals", str(d + i)),
						                    Xi=thread[i].Xi, Yi=thread[i].Yi,
						                    Ni_X=thread[i].Ni_X, Ni_Y=thread[i].Ni_Y)
						sp.save_npz(os.path.join(self.directory, "Ni_XY", str(d + i)), thread[i].Ni_XY)
						sp.save_npz(os.path.join(self.directory, "pmi", str(d + i)), thread[i].pmi)
						distances_computed += 1
						pbar.update(1)

					d += self.no_of_threads

		except KeyboardInterrupt:
			print(f"\nInterrupted at d={d}")

		return distances_computed
