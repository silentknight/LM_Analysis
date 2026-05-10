#!/usr/bin/env python

# System libs
import numpy as np
import sys
import threading
import os
import scipy.sparse

class myThread(threading.Thread):
	def __init__(self, wordID, dataArrayLen):
		self.wordID = wordID
		self.dataArrayLen = dataArrayLen
		self.recurrenceLengthAndFrequency = []
		self.complete = False
		super(myThread, self).__init__()

	def run(self):
		indexes = np.where(dataArray==self.wordID)
		if uniqueElementsFrequency[self.wordID] == len(indexes[0]):
			tempA = indexes[0][0:len(indexes[0])-1]
			tempB = indexes[0][1:]
			temp = np.unique(tempB-tempA,return_counts=True)
			self.recurrenceLengthAndFrequency = np.array(temp)
		self.complete = True

class Recurrence(object):
	def __init__(self, corpusData, no_of_threads):
		global dataArray
		global uniqueElementsFrequency
		dataArray = np.asarray(corpusData.sequentialData.dataArray, dtype=np.uint64)
		uniqueElementsFrequency = corpusData.dictionary.counter
		self.uniqueSize = corpusData.dictionary.totalUnique
		self.dataArrayLen = dataArray.size
		self.no_of_threads = no_of_threads
		self.recurrenceList = self.getRecurrence()

	def getRecurrence(self):
		no_of_threads = self.no_of_threads
		recurrenceList = {}
		index = 0

		try:
			while index<self.uniqueSize:
				if self.uniqueSize-index < self.no_of_threads:
					no_of_threads = self.uniqueSize-index

				thread = []
				for i in range(no_of_threads):
					thread.append(myThread(index+i,self.dataArrayLen))

				for i in range(no_of_threads):
					thread[i].start()

				for i in range(no_of_threads):
					thread[i].join()

				for i in range(no_of_threads):
					recurrenceList[index+i] = thread[i].recurrenceLengthAndFrequency

				if float(index+1+i)/float(self.uniqueSize)*100%10 == 0:
					print("Processed: {0}%".format(float(index+1+i)/float(self.uniqueSize)*100))

				index += self.no_of_threads

		except KeyboardInterrupt:
			print("Processed upto: {0}".format(str(index+i)))

		return recurrenceList
