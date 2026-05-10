#!/usr/bin/env python

import numpy as np
import scipy.special as spec
import sys
import threading
import os
import array
import scipy.sparse
import speedup


class myThread(threading.Thread):
	def __init__(self, d, overlap, log_type, method, data_array, line_length_list, total_length):
		self.d = d
		self.mi = 0
		self.Hx = 0
		self.Hy = 0
		self.Hxy = 0
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

		log = lambda val, base: np.log(val) if base == 0 else (np.log2(val) if base == 1 else np.log10(val))

		Ni_X = Ni_X.data
		Ni_Y = Ni_Y.data
		Ni_XY = Ni_XY.data

		if self.method == "grassberger":
			self.Hx = log(np.sum(Ni_X), self.log_type) - np.sum(Ni_X * spec.digamma(Ni_X)) / np.sum(Ni_X)
			self.Hy = log(np.sum(Ni_Y), self.log_type) - np.sum(Ni_Y * spec.digamma(Ni_Y)) / np.sum(Ni_Y)
			self.Hxy = log(np.sum(Ni_XY), self.log_type) - np.sum(Ni_XY * spec.digamma(Ni_XY)) / np.sum(Ni_XY)
			self.mi = self.Hx + self.Hy - self.Hxy
		elif self.method == "standard":
			self.Hx = -1 * np.sum(Ni_X / np.sum(Ni_X) * log(Ni_X / np.sum(Ni_X), self.log_type))
			self.Hy = -1 * np.sum(Ni_Y / np.sum(Ni_Y) * log(Ni_Y / np.sum(Ni_Y), self.log_type))
			self.Hxy = -1 * np.sum(Ni_XY / np.sum(Ni_XY) * log(Ni_XY / np.sum(Ni_XY), self.log_type))
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
		mi = np.zeros(0)
		Hx = np.zeros(0)
		Hy = np.zeros(0)
		Hxy = np.zeros(0)
		d = 1

		print("Average String Length: ", int(self.corpus.sequentialData.averageLength))
		print("Total String Length: ", int(self.corpus.sequentialData.totalLength))

		# Load previously computed distances if the file exists
		if os.path.exists(self.filename):
			with open(self.filename, "r") as f:
				lines = f.readlines()

			if lines:
				temp = lines[0].split()
				if len(temp) >= 2 and temp[0] == "data:" and temp[1] == self.corpus.datainfo:
					for line in lines:
						temp = line.strip().split(":")
						if temp[0] == "d":
							try:
								temp1 = temp[2].split(",")
								mi = np.append(mi, np.zeros(1))
								mi[int(temp[1]) - 1] = float(temp1[0])
								Hx = np.append(Hx, np.zeros(1))
								Hx[int(temp[1]) - 1] = float(temp1[1])
								Hy = np.append(Hy, np.zeros(1))
								Hy[int(temp[1]) - 1] = float(temp1[2])
								Hxy = np.append(Hxy, np.zeros(1))
								Hxy[int(temp[1]) - 1] = float(temp1[3])
								d = int(temp[1]) + 1
							except (IndexError, ValueError):
								print("Warning: skipping malformed line: " + line.strip())
		else:
			print("File does not exist to load previous data")

		end = False

		# Atomically rewrite existing data so the file is not left half-truncated on crash
		tmp_path = self.filename + '.tmp'
		with open(tmp_path, 'w') as f_tmp:
			f_tmp.write("data: " + self.corpus.datainfo + "\n")
			for i in range(len(mi)):
				f_tmp.write("d:" + str(i + 1) + ":" + str(mi[i]) + "," +
				            str(Hx[i]) + "," + str(Hy[i]) + "," + str(Hxy[i]) + "\n")
		os.replace(tmp_path, self.filename)

		f = open(self.filename, 'a')

		try:
			max_distance = self.total_length
			while d < max_distance and d <= self.cutoff and not end:
				mi = np.append(mi, np.zeros(self.no_of_threads))
				Hx = np.append(Hx, np.zeros(self.no_of_threads))
				Hy = np.append(Hy, np.zeros(self.no_of_threads))
				Hxy = np.append(Hxy, np.zeros(self.no_of_threads))

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
						# Trim trailing zero slots for threads that did not compute
						threads_remaining = self.no_of_threads - i
						for _ in range(threads_remaining):
							if len(mi) > 0 and mi[-1] == 0:
								mi = np.delete(mi, -1)
								Hx = np.delete(Hx, -1)
								Hy = np.delete(Hy, -1)
								Hxy = np.delete(Hxy, -1)
						break

					mi[d + i - 1] = thread[i].mi
					Hx[d + i - 1] = thread[i].Hx
					Hy[d + i - 1] = thread[i].Hy
					Hxy[d + i - 1] = thread[i].Hxy

					print(thread[i].d, thread[i].mi, thread[i].Hx, thread[i].Hy,
					      thread[i].Hx + thread[i].Hy, thread[i].Hxy)
					f.write("d:" + str(d + i) + ":" + str(mi[d + i - 1]) + "," +
					        str(Hx[d + i - 1]) + "," + str(Hy[d + i - 1]) + "," +
					        str(Hxy[d + i - 1]) + "\n")

				d += self.no_of_threads

		except KeyboardInterrupt:
			print("Processed upto: " + str(d - 1))

		f.close()

		return mi, Hx, Hy, Hxy
