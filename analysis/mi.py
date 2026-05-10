#!/usr/bin/env python

import csv
import numpy as np
import scipy.special as spec
import sys
import threading
import os
import array
import scipy.sparse
from . import speedup


class myThread(threading.Thread):
	def __init__(self, d, overlap, log_type, method, data_array, line_length_list, total_length):
		self.d = d
		self.mi = 0.0
		self.Hx = 0.0
		self.Hy = 0.0
		self.Hxy = 0.0
		self.complete = False
		self.overlap = overlap
		self.method = method
		self.log_type = log_type
		self.data_array = data_array
		self.line_length_list = line_length_list
		self.total_length = total_length
		super(myThread, self).__init__()

	def run(self):
		Ni_X, Ni_Y, Ni_XY, u_X, u_Y = speedup.getJointRV(
			self.data_array, self.line_length_list, self.total_length, self.d, self.overlap)

		if Ni_X is None:
			self.complete = True
			return

		# Ni_X and Ni_Y are dense float64 arrays from getJointRV;
		# Ni_XY is a CSC sparse matrix — extract non-zero counts
		Ni_XY = Ni_XY.data

		log = lambda val, base: np.log(val) if base == 0 else (np.log2(val) if base == 1 else np.log10(val))

		if self.method == "grassberger":
			sum_X = np.sum(Ni_X)
			sum_Y = np.sum(Ni_Y)
			sum_XY = np.sum(Ni_XY)
			self.Hx = log(sum_X, self.log_type) - np.sum(Ni_X * spec.digamma(Ni_X)) / sum_X
			self.Hy = log(sum_Y, self.log_type) - np.sum(Ni_Y * spec.digamma(Ni_Y)) / sum_Y
			self.Hxy = log(sum_XY, self.log_type) - np.sum(Ni_XY * spec.digamma(Ni_XY)) / sum_XY
			self.mi = self.Hx + self.Hy - self.Hxy
		elif self.method == "standard":
			px = Ni_X / np.sum(Ni_X)
			py = Ni_Y / np.sum(Ni_Y)
			pxy = Ni_XY / np.sum(Ni_XY)
			self.Hx = -np.sum(px * log(px, self.log_type))
			self.Hy = -np.sum(py * log(py, self.log_type))
			self.Hxy = -np.sum(pxy * log(pxy, self.log_type))
			self.mi = self.Hx + self.Hy - self.Hxy


class MutualInformation(object):
	def __init__(self, corpusData, log_type, no_of_threads, data_file_path, overlap, method, cutoff):
		self.corpus = corpusData
		self.data_array = array.array('L', corpusData.sequentialData.dataArray)
		self.line_length_list = np.array(corpusData.sequentialData.wordCountList, dtype=np.uint64)
		self.total_length = corpusData.sequentialData.totalLength
		self.no_of_threads = no_of_threads
		self.filename = data_file_path
		self.overlap = overlap
		self.method = method
		self.log_type = log_type
		self.cutoff = cutoff
		self.mutualInformation = self.calculate_MI()

	def calculate_MI(self):
		# Use Python lists for O(1) amortized append; convert to numpy at the end.
		# np.append in a loop is O(n) per call → O(n²) over the full computation.
		mi, Hx, Hy, Hxy = [], [], [], []
		d = 1

		print("Average String Length: ", int(self.corpus.sequentialData.averageLength))
		print("Total String Length: ", int(self.corpus.sequentialData.totalLength))

		# Load previously computed distances if the file exists
		if os.path.exists(self.filename):
			with open(self.filename, "r", newline='') as f:
				reader = csv.reader(f)
				try:
					header = next(reader)
					if len(header) >= 2 and header[0] == "data" and header[1] == self.corpus.datainfo:
						next(reader)  # skip column header row
						for row in reader:
							if len(row) < 5:
								continue
							try:
								mi.append(float(row[1]))
								Hx.append(float(row[2]))
								Hy.append(float(row[3]))
								Hxy.append(float(row[4]))
								d = int(row[0]) + 1
							except (IndexError, ValueError):
								print("Warning: skipping malformed row: " + str(row))
				except StopIteration:
					pass
		else:
			print("File does not exist to load previous data")

		end = False

		# Atomically rewrite existing data so the file is not left half-truncated on crash
		tmp_path = self.filename + '.tmp'
		with open(tmp_path, 'w', newline='') as f_tmp:
			w = csv.writer(f_tmp)
			w.writerow(["data", self.corpus.datainfo])
			w.writerow(["d", "mi", "Hx", "Hy", "Hxy"])
			for i in range(len(mi)):
				w.writerow([i + 1, mi[i], Hx[i], Hy[i], Hxy[i]])
		os.replace(tmp_path, self.filename)

		f = open(self.filename, 'a', newline='')

		try:
			max_distance = self.total_length
			while d < max_distance and d <= self.cutoff and not end:
				# Pre-extend lists with placeholder zeros for this batch
				mi.extend([0.0] * self.no_of_threads)
				Hx.extend([0.0] * self.no_of_threads)
				Hy.extend([0.0] * self.no_of_threads)
				Hxy.extend([0.0] * self.no_of_threads)

				thread = []
				for i in range(self.no_of_threads):
					thread.append(myThread(
						d + i, self.overlap, self.log_type, self.method,
						self.data_array, self.line_length_list, self.total_length))

				for i in range(self.no_of_threads):
					thread[i].start()

				for i in range(self.no_of_threads):
					thread[i].join()

				for i in range(self.no_of_threads):
					if thread[i].complete or np.isnan(thread[i].mi):
						end = True
						# Trim trailing placeholder slots (O(1) list pop vs O(n) np.delete)
						threads_remaining = self.no_of_threads - i
						for _ in range(threads_remaining):
							if mi and mi[-1] == 0.0:
								del mi[-1]; del Hx[-1]; del Hy[-1]; del Hxy[-1]
						break

					mi[d + i - 1] = thread[i].mi
					Hx[d + i - 1] = thread[i].Hx
					Hy[d + i - 1] = thread[i].Hy
					Hxy[d + i - 1] = thread[i].Hxy

					print(thread[i].d, thread[i].mi, thread[i].Hx, thread[i].Hy,
					      thread[i].Hx + thread[i].Hy, thread[i].Hxy)
					csv.writer(f).writerow(
					    [d + i, mi[d + i - 1], Hx[d + i - 1], Hy[d + i - 1], Hxy[d + i - 1]])

				d += self.no_of_threads

		except KeyboardInterrupt:
			print("Processed upto: " + str(d - 1))

		f.close()

		return np.array(mi), np.array(Hx), np.array(Hy), np.array(Hxy)
