#!/usr/bin/env python

import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor


class Recurrence(object):
	def __init__(self, corpusData, no_of_threads):
		self.uniqueSize = corpusData.dictionary.totalUnique
		self.no_of_threads = no_of_threads

		# Build position index using argsort — O(N log N) entirely in C.
		# The previous approach used a Python for-loop over N tokens which was
		# ~10-50x slower on large corpora due to per-element Python overhead.
		data = np.asarray(corpusData.sequentialData.dataArray, dtype=np.int64)
		sort_idx = np.argsort(data, kind='stable')  # corpus positions sorted by word ID
		sorted_ids = data[sort_idx]
		splits = np.flatnonzero(np.diff(sorted_ids)) + 1
		groups = np.split(sort_idx, splits)
		# first element of each group is the word ID it belongs to
		unique_ids = sorted_ids[np.concatenate(([0], splits))]

		self.position_index = [np.empty(0, dtype=np.uint64)] * self.uniqueSize
		for word_id, group in zip(unique_ids, groups):
			self.position_index[word_id] = np.sort(group).astype(np.uint64)

		self.recurrenceList = self.getRecurrence()

	def getRecurrence(self):
		report_interval = max(1, self.uniqueSize // 10)

		def compute(positions):
			if len(positions) > 1:
				gaps = np.diff(positions)
				return np.array(np.unique(gaps, return_counts=True))
			return []

		recurrenceList = {}
		try:
			# ThreadPoolExecutor reuses a fixed pool — avoids creating and destroying
			# O(V) Thread objects (one per word in the vocabulary).
			# np.diff / np.unique release the GIL, so threads run in parallel.
			with ThreadPoolExecutor(max_workers=self.no_of_threads) as executor:
				for i, result in enumerate(executor.map(compute, self.position_index)):
					recurrenceList[i] = result
					if (i + 1) % report_interval == 0:
						print("Processed: {:.0f}%".format((i + 1) * 100.0 / self.uniqueSize))
		except KeyboardInterrupt:
			print("Processed upto: {0}".format(len(recurrenceList)))

		return recurrenceList
