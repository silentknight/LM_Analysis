#!/usr/bin/env python

# System libs
import os
import sys
from collections import Counter
import simplejson as json
import gzip
import pickle


class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []
		self.counter = Counter()
		self.totalUnique = 0

	def add_word(self, word):
		word = str(word)
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
			self.totalUnique += 1
		token_id = self.word2idx[word]
		self.counter[token_id] += 1
		return self.word2idx[word]

	def get_word_from_id(self, wordID):
		return self.idx2word[wordID]

	def __len__(self):
		return len(self.idx2word)


class SequentialData(object):
	def __init__(self):
		self.__wordLine = []
		self.dataArray = []
		self.wordCountList = []
		self.fileStartIndex = []
		self.averageLength = 0
		self.totalLength = 0

	def add_to_list(self, wordID):
		self.__wordLine.append(wordID)

	def add_data(self):
		self.dataArray = self.dataArray + self.__wordLine
		self.averageLength = (self.averageLength * len(self.wordCountList) + len(self.__wordLine)) / (len(self.wordCountList) + 1)
		self.wordCountList.append(len(self.__wordLine))
		self.fileStartIndex.append(self.totalLength)
		self.totalLength += len(self.__wordLine)
		self.__wordLine = []


class Corpus(object):
	def __init__(self, path, ifwords):
		self.dictionary = Dictionary()
		self.sequentialData = SequentialData()
		self.datainfo = path
		self.ifwords = ifwords
		self.completion = self.choose_dataset(self.datainfo)
		print("Total Length: {0}".format(self.sequentialData.totalLength))

	def choose_dataset(self, path):

		if path == "dataset/ptb/.full":
			print("Penn Tree Bank Full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/ptb/.train":
			print("Penn Tree Bank Train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/ptb/.valid":
			print("Penn Tree Bank Valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
		elif path == "dataset/ptb/.test":
			print("Penn Tree Bank Test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test'))


		elif path == "dataset/wiki2/.full":
			print("wikitext-2 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/wiki2/.train":
			print("wikitext-2 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/wiki2/.valid":
			print("wikitext-2 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
		elif path == "dataset/wiki2/.test":
			print("wikitext-2 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test'))

		elif path == "dataset/wiki2R/.full":
			print("wikitext-2 raw full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/wiki2R/.train":
			print("wikitext-2 raw train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/wiki2R/.valid":
			print("wikitext-2 raw valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
		elif path == "dataset/wiki2R/.test":
			print("wikitext-2 raw test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test'))

		elif path == "dataset/wiki2C/.full":
			print("wikitext-2 cleaned full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainC'))
			self.valid = self.tokenize_file(os.path.join(path, 'validC'))
			self.test = self.tokenize_file(os.path.join(path, 'testC'))
		elif path == "dataset/wiki2C/.test":
			print("wikitext-2 cleaned testC dataset")
			path = path.split('.')[0]
			self.testC = self.tokenize_file(os.path.join(path, 'testC'))
		elif path == "dataset/wiki2C/.test1":
			print("wikitext-2 cleaned testC1 dataset")
			path = path.split('.')[0]
			self.testC1 = self.tokenize_file(os.path.join(path, 'testC1'))
		elif path == "dataset/wiki2C/.test2":
			print("wikitext-2 cleaned testC2 dataset")
			path = path.split('.')[0]
			self.testC2 = self.tokenize_file(os.path.join(path, 'testC2'))
		elif path == "dataset/wiki2C/.train":
			print("wikitext-2 cleaned train dataset")
			path = path.split('.')[0]
			self.trainC = self.tokenize_file(os.path.join(path, 'trainC'))
		elif path == "dataset/wiki2C/.valid":
			print("wikitext-2 cleaned validC dataset")
			path = path.split('.')[0]
			self.validC = self.tokenize_file(os.path.join(path, 'validC'))
		elif path == "dataset/wiki2C/.valid1":
			print("wikitext-2 cleaned validC1 dataset")
			path = path.split('.')[0]
			self.validC1 = self.tokenize_file(os.path.join(path, 'validC1'))


		elif path == "dataset/wiki103/.full":
			print("wikitext-103 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/wiki103/.train":
			print("wikitext-103 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/wiki103/.valid":
			print("wikitext-103 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
		elif path == "dataset/wiki103/.test":
			print("wikitext-103 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test'))

		elif path == "dataset/wiki103C/.full":
			print("wikitext-103 cleaned full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainC'))
			self.valid = self.tokenize_file(os.path.join(path, 'validC'))
			self.test = self.tokenize_file(os.path.join(path, 'testC'))
		elif path == "dataset/wiki103C/.train":
			print("wikitext-103 cleaned trainC dataset")
			path = path.split('.')[0]
			self.trainC = self.tokenize_file(os.path.join(path, 'trainC'))
		elif path == "dataset/wiki103C/.test":
			print("wikitext-103 cleaned testC dataset")
			path = path.split('.')[0]
			self.testC = self.tokenize_file(os.path.join(path, 'testC'))
		elif path == "dataset/wiki103C/.valid":
			print("wikitext-103 cleaned validC dataset")
			path = path.split('.')[0]
			self.validC = self.tokenize_file(os.path.join(path, 'validC'))

		elif path == "dataset/wiki103R/.full":
			print("wikitext-103 raw full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/wiki103R/.train":
			print("wikitext-103 raw train dataset")
			path = path.split('.')[0]
			self.trainC = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/wiki103R/.test":
			print("wikitext-103 raw test dataset")
			path = path.split('.')[0]
			self.testC = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/wiki103R/.valid":
			print("wikitext-103 raw valid dataset")
			path = path.split('.')[0]
			self.validC = self.tokenize_file(os.path.join(path, 'valid'))


		elif path == "dataset/wiki19/.full":
			print("wikitext-19 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/wiki19/.train":
			print("wikitext-19 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/wiki19/.valid":
			print("wikitext-19 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
		elif path == "dataset/wiki19/.test":
			print("wikitext-19 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test'))

		elif path == "dataset/wiki19C/.full":
			print("wikitext-19 cleaned full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/wiki19C/.train":
			print("wikitext-19 cleaned train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/wiki19C/.valid":
			print("wikitext-19 cleaned valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
		elif path == "dataset/wiki19C/.test":
			print("wikitext-19 cleaned test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test'))

		elif path == "dataset/wiki19L/.full":
			print("wikitext-19L Text8-Like full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/wiki19L/.train":
			print("wikitext-19L Text8-Like train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/wiki19L/.valid":
			print("wikitext-19L Text8-Like valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
		elif path == "dataset/wiki19L/.test":
			print("wikitext-19L Text8-Like test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test'))

		elif path == "dataset/wiki19L-wor/.full":
			print("wikitext-19L-wor Text8-Like wor full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/wiki19L-wor/.train":
			print("wikitext-19L-wor Text-Like wor train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/wiki19L-wor/.valid":
			print("wikitext-19L-wor Text-Like wor valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
		elif path == "dataset/wiki19L-wor/.test":
			print("wikitext-19L-wor Text8-Like wor test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test'))


		elif path == "dataset/wiki2M/.test1":
			print("wikitext-2 test1 dataset")
			path = path.split('.')[0]
			self.test1 = self.tokenize_file(os.path.join(path, 'test1'))
		elif path == "dataset/wiki2M/.test2":
			print("wikitext-2 test2 dataset")
			path = path.split('.')[0]
			self.test2 = self.tokenize_file(os.path.join(path, 'test2'))
		elif path == "dataset/wiki2M/.valid1":
			print("wikitext-2 valid1 dataset")
			path = path.split('.')[0]
			self.valid1 = self.tokenize_file(os.path.join(path, 'valid1'))


		elif path == "dataset/wiki2Resample/.full1":
			print("wikitext-2 Reshaped 1 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainR1'))
			self.valid = self.tokenize_file(os.path.join(path, 'validR1'))
			self.test = self.tokenize_file(os.path.join(path, 'testR1'))
		elif path == "dataset/wiki2Resample/.test1":
			print("wikitext-2 Reshaped 1 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'testR1'))
		elif path == "dataset/wiki2Resample/.train1":
			print("wikitext-2 Reshaped 1 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainR1'))
		elif path == "dataset/wiki2Resample/.valid1":
			print("wikitext-2 Reshaped 1 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'validR1'))
		elif path == "dataset/wiki2Resample/.full2":
			print("wikitext-2 Reshaped 2 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainR2'))
			self.valid = self.tokenize_file(os.path.join(path, 'validR2'))
			self.test = self.tokenize_file(os.path.join(path, 'testR2'))
		elif path == "dataset/wiki2Resample/.test2":
			print("wikitext-2 Reshaped 2 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'testR2'))
		elif path == "dataset/wiki2Resample/.train2":
			print("wikitext-2 Reshaped 2 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainR2'))
		elif path == "dataset/wiki2Resample/.valid2":
			print("wikitext-2 Reshaped 2 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'validR2'))
		elif path == "dataset/wiki2Resample/.full3":
			print("wikitext-2 Reshaped 3 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainR3'))
			self.valid = self.tokenize_file(os.path.join(path, 'validR3'))
			self.test = self.tokenize_file(os.path.join(path, 'testR3'))
		elif path == "dataset/wiki2Resample/.test3":
			print("wikitext-2 Reshaped 3 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'testR3'))
		elif path == "dataset/wiki2Resample/.train3":
			print("wikitext-2 Reshaped 3 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainR3'))
		elif path == "dataset/wiki2Resample/.valid3":
			print("wikitext-2 Reshaped 3 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'validR3'))
		elif path == "dataset/wiki2Resample/.full4":
			print("wikitext-2 Reshaped 4 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainR4'))
			self.valid = self.tokenize_file(os.path.join(path, 'validR4'))
			self.test = self.tokenize_file(os.path.join(path, 'testR4'))
		elif path == "dataset/wiki2Resample/.test4":
			print("wikitext-2 Reshaped 4 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'testR4'))
		elif path == "dataset/wiki2Resample/.train4":
			print("wikitext-2 Reshaped 4 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainR4'))
		elif path == "dataset/wiki2Resample/.valid4":
			print("wikitext-2 Reshaped 4 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'validR4'))


		elif path == "dataset/wiki2Samples/.full1":
			print("wikitext-2 Samples 1 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train_1'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid_1'))
			self.test = self.tokenize_file(os.path.join(path, 'test_1'))
		elif path == "dataset/wiki2Samples/.test1":
			print("wikitext-2 Samples 1 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test_1'))
		elif path == "dataset/wiki2Samples/.train1":
			print("wikitext-2 Samples 1 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train_1'))
		elif path == "dataset/wiki2Samples/.valid1":
			print("wikitext-2 Samples 1 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid_1'))
		elif path == "dataset/wiki2Samples/.full2":
			print("wikitext-2 Samples 2 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train_2'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid_2'))
			self.test = self.tokenize_file(os.path.join(path, 'test_2'))
		elif path == "dataset/wiki2Samples/.test2":
			print("wikitext-2 Samples 2 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test_2'))
		elif path == "dataset/wiki2Samples/.train2":
			print("wikitext-2 Samples 2 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train_2'))
		elif path == "dataset/wiki2Samples/.valid2":
			print("wikitext-2 Samples 2 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid_2'))
		elif path == "dataset/wiki2Samples/.full3":
			print("wikitext-2 Samples 3 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train_3'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid_3'))
			self.test = self.tokenize_file(os.path.join(path, 'test_3'))
		elif path == "dataset/wiki2Samples/.test3":
			print("wikitext-2 Samples 3 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test_3'))
		elif path == "dataset/wiki2Samples/.train3":
			print("wikitext-2 Samples 3 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train_3'))
		elif path == "dataset/wiki2Samples/.valid3":
			print("wikitext-2 Samples 3 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid_3'))
		elif path == "dataset/wiki2Samples/.full4":
			print("wikitext-2 Samples 4 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train_4'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid_4'))
			self.test = self.tokenize_file(os.path.join(path, 'test_4'))
		elif path == "dataset/wiki2Samples/.test4":
			print("wikitext-2 Samples 4 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test_4'))
		elif path == "dataset/wiki2Samples/.train4":
			print("wikitext-2 Samples 4 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train_4'))
		elif path == "dataset/wiki2Samples/.valid4":
			print("wikitext-2 Samples 4 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid_4'))


		elif path == "dataset/wiki2Homogenous/.fullH1":
			print("wikitext-2 Homogenous 1 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainH1'))
			self.valid = self.tokenize_file(os.path.join(path, 'validH1'))
			self.test = self.tokenize_file(os.path.join(path, 'testH1'))
		elif path == "dataset/wiki2Homogenous/.testH1":
			print("wikitext-2 Homogenous 1 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'testH1'))
		elif path == "dataset/wiki2Homogenous/.trainH1":
			print("wikitext-2 Homogenous 1 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainH1'))
		elif path == "dataset/wiki2Homogenous/.validH1":
			print("wikitext-2 Homogenous 1 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'validH1'))
		elif path == "dataset/wiki2Homogenous/.fullH2":
			print("wikitext-2 Homogenous 2 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainH2'))
			self.valid = self.tokenize_file(os.path.join(path, 'validH2'))
			self.test = self.tokenize_file(os.path.join(path, 'testH2'))
		elif path == "dataset/wiki2Homogenous/.testH2":
			print("wikitext-2 Homogenous 2 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'testH2'))
		elif path == "dataset/wiki2Homogenous/.trainH2":
			print("wikitext-2 Homogenous 2 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainH2'))
		elif path == "dataset/wiki2Homogenous/.validH2":
			print("wikitext-2 Homogenous 2 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'validH2'))
		elif path == "dataset/wiki2Homogenous/.fullCH1":
			print("wikitext-2 Homogenous Cleaned 1 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainCH1'))
			self.valid = self.tokenize_file(os.path.join(path, 'validCH1'))
			self.test = self.tokenize_file(os.path.join(path, 'testCH1'))
		elif path == "dataset/wiki2Homogenous/.testCH1":
			print("wikitext-2 Homogenous Cleaned 1 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'testCH1'))
		elif path == "dataset/wiki2Homogenous/.trainCH1":
			print("wikitext-2 Homogenous Cleaned 1 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainCH1'))
		elif path == "dataset/wiki2Homogenous/.validCH1":
			print("wikitext-2 Homogenous Cleaned 1 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'validCH1'))
		elif path == "dataset/wiki2Homogenous/.fullCH2":
			print("wikitext-2 Homogenous Cleaned 2 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainCH2'))
			self.valid = self.tokenize_file(os.path.join(path, 'validCH2'))
			self.test = self.tokenize_file(os.path.join(path, 'testCH2'))
		elif path == "dataset/wiki2Homogenous/.testCH2":
			print("wikitext-2 Homogenous Cleaned 2 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'testCH2'))
		elif path == "dataset/wiki2Homogenous/.trainCH2":
			print("wikitext-2 Homogenous Cleaned 2 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'trainCH2'))
		elif path == "dataset/wiki2Homogenous/.validCH2":
			print("wikitext-2 Homogenous Cleaned 2 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'validCH2'))


		elif path == "dataset/hutter/enwik8":
			print("Hutter Enwik8 dataset")
			self.enwik8 = self.tokenize_file(path)
		elif path == "dataset/hutter/text8":
			print("Hutter Text8 dataset")
			self.text8 = self.tokenize_file(path)
		elif path == "dataset/hutter/text8-small":
			print("text8 small dataset")
			self.train = self.tokenize_file(path)
		elif path == "dataset/hutter/text8-wo-r-2":
			print("text8 without rare words 2")
			self.full = self.tokenize_file(path)
		elif path == "dataset/hutter/text8-small-wo-r-2":
			print("text8 small without rare words 2")
			self.full = self.tokenize_file(path)
		elif path == "dataset/hutter/text8-small-wo-r-4":
			print("text8 small without rare words 4")
			self.full = self.tokenize_file(path)


		elif path == "dataset/hutter/text8w/.full":
			print("text8w full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/hutter/text8w/.train":
			print("text8w train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/hutter/text8w/.valid":
			print("text8w valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
		elif path == "dataset/hutter/text8w/.test":
			print("text8w test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test'))


		elif path == "dataset/hutter/text8w_S/.full":
			print("text8w full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
			self.test = self.tokenize_file(os.path.join(path, 'test'))
		elif path == "dataset/hutter/text8w_S/.train":
			print("text8w train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train'))
		elif path == "dataset/hutter/text8w_S/.valid":
			print("text8w valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid'))
		elif path == "dataset/hutter/text8w_S/.test":
			print("text8w test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test'))


		elif path == "dataset/hutter/text8w_small/.full":
			print("text8w small full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train.txt'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid.txt'))
			self.test = self.tokenize_file(os.path.join(path, 'test.txt'))
		elif path == "dataset/hutter/text8w_small/.train":
			print("text8w small train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train.txt'))
		elif path == "dataset/hutter/text8w_small/.valid":
			print("text8w small valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid.txt'))
		elif path == "dataset/hutter/text8w_small/.test":
			print("text8w small test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test.txt'))


		elif path == "dataset/ptb_text8/.full":
			print("ptb_text8 full dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train.txt'))
			self.valid = self.tokenize_file(os.path.join(path, 'valid.txt'))
			self.test = self.tokenize_file(os.path.join(path, 'test.txt'))
		elif path == "dataset/ptb_text8/.train":
			print("ptb_text8 train dataset")
			path = path.split('.')[0]
			self.train = self.tokenize_file(os.path.join(path, 'train.txt'))
		elif path == "dataset/ptb_text8/.valid":
			print("ptb_text8 valid dataset")
			path = path.split('.')[0]
			self.valid = self.tokenize_file(os.path.join(path, 'valid.txt'))
		elif path == "dataset/ptb_text8/.test":
			print("ptb_text8 test dataset")
			path = path.split('.')[0]
			self.test = self.tokenize_file(os.path.join(path, 'test.txt'))
		elif path == "dataset/ptb_text8/train_small":
			print("ptb_text8 train dataset")
			self.train = self.tokenize_file(path)

		else:
			print("Please check the dataset path supplied. No such path found")
			sys.exit(0)
		return 1

	def tokenize_file(self, path):
		assert os.path.exists(path)
		print(path)
		with open(path, 'r') as f:
			tokens = 0
			for line in f:
				if self.ifwords == 1:
					words = line.split() + ['<eos>']
				else:
					words = list(line.strip().replace("<unk>", "^")) + [" "]

				tokens += len(words)
				for word in words:
					wordID = self.dictionary.add_word(word)
					self.sequentialData.add_to_list(wordID)
		self.sequentialData.add_data()
		print("Size of Vocabulary: {0}".format(self.dictionary.totalUnique))
