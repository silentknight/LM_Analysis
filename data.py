#!/usr/bin/env python

import os
import sys
from collections import Counter


# Registry: path_key → (label, is_subdir, [(attr, filename), ...])
# is_subdir=True  → base = key.rsplit('.', 1)[0]; files resolved relative to base
# is_subdir=False → file is the key itself; attr is the destination attribute name
_REGISTRY = {
	# Penn Tree Bank
	"dataset/ptb/.full":  ("Penn Tree Bank Full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/ptb/.train": ("Penn Tree Bank Train", True, [("train","train")]),
	"dataset/ptb/.valid": ("Penn Tree Bank Valid", True, [("valid","valid")]),
	"dataset/ptb/.test":  ("Penn Tree Bank Test",  True, [("test","test")]),

	# wikitext-2
	"dataset/wiki2/.full":  ("wikitext-2 full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/wiki2/.train": ("wikitext-2 train", True, [("train","train")]),
	"dataset/wiki2/.valid": ("wikitext-2 valid", True, [("valid","valid")]),
	"dataset/wiki2/.test":  ("wikitext-2 test",  True, [("test","test")]),

	# wikitext-2 raw
	"dataset/wiki2R/.full":  ("wikitext-2 raw full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/wiki2R/.train": ("wikitext-2 raw train", True, [("train","train")]),
	"dataset/wiki2R/.valid": ("wikitext-2 raw valid", True, [("valid","valid")]),
	"dataset/wiki2R/.test":  ("wikitext-2 raw test",  True, [("test","test")]),

	# wikitext-2 cleaned
	"dataset/wiki2C/.full":   ("wikitext-2 cleaned full",   True, [("train","trainC"), ("valid","validC"), ("test","testC")]),
	"dataset/wiki2C/.test":   ("wikitext-2 cleaned test",   True, [("testC","testC")]),
	"dataset/wiki2C/.test1":  ("wikitext-2 cleaned test1",  True, [("testC1","testC1")]),
	"dataset/wiki2C/.test2":  ("wikitext-2 cleaned test2",  True, [("testC2","testC2")]),
	"dataset/wiki2C/.train":  ("wikitext-2 cleaned train",  True, [("trainC","trainC")]),
	"dataset/wiki2C/.valid":  ("wikitext-2 cleaned valid",  True, [("validC","validC")]),
	"dataset/wiki2C/.valid1": ("wikitext-2 cleaned valid1", True, [("validC1","validC1")]),

	# wikitext-103
	"dataset/wiki103/.full":  ("wikitext-103 full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/wiki103/.train": ("wikitext-103 train", True, [("train","train")]),
	"dataset/wiki103/.valid": ("wikitext-103 valid", True, [("valid","valid")]),
	"dataset/wiki103/.test":  ("wikitext-103 test",  True, [("test","test")]),

	# wikitext-103 cleaned
	"dataset/wiki103C/.full":  ("wikitext-103 cleaned full",  True, [("train","trainC"), ("valid","validC"), ("test","testC")]),
	"dataset/wiki103C/.train": ("wikitext-103 cleaned train", True, [("trainC","trainC")]),
	"dataset/wiki103C/.test":  ("wikitext-103 cleaned test",  True, [("testC","testC")]),
	"dataset/wiki103C/.valid": ("wikitext-103 cleaned valid", True, [("validC","validC")]),

	# wikitext-103 raw (was incorrectly assigning to trainC/testC/validC; fixed to train/test/valid)
	"dataset/wiki103R/.full":  ("wikitext-103 raw full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/wiki103R/.train": ("wikitext-103 raw train", True, [("train","train")]),
	"dataset/wiki103R/.test":  ("wikitext-103 raw test",  True, [("test","test")]),
	"dataset/wiki103R/.valid": ("wikitext-103 raw valid", True, [("valid","valid")]),

	# wikitext-19
	"dataset/wiki19/.full":  ("wikitext-19 full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/wiki19/.train": ("wikitext-19 train", True, [("train","train")]),
	"dataset/wiki19/.valid": ("wikitext-19 valid", True, [("valid","valid")]),
	"dataset/wiki19/.test":  ("wikitext-19 test",  True, [("test","test")]),

	# wikitext-19 cleaned
	"dataset/wiki19C/.full":  ("wikitext-19 cleaned full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/wiki19C/.train": ("wikitext-19 cleaned train", True, [("train","train")]),
	"dataset/wiki19C/.valid": ("wikitext-19 cleaned valid", True, [("valid","valid")]),
	"dataset/wiki19C/.test":  ("wikitext-19 cleaned test",  True, [("test","test")]),

	# wikitext-19L (Text8-Like)
	"dataset/wiki19L/.full":  ("wikitext-19L Text8-Like full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/wiki19L/.train": ("wikitext-19L Text8-Like train", True, [("train","train")]),
	"dataset/wiki19L/.valid": ("wikitext-19L Text8-Like valid", True, [("valid","valid")]),
	"dataset/wiki19L/.test":  ("wikitext-19L Text8-Like test",  True, [("test","test")]),

	# wikitext-19L-wor (Text8-Like without rare words)
	"dataset/wiki19L-wor/.full":  ("wikitext-19L-wor full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/wiki19L-wor/.train": ("wikitext-19L-wor train", True, [("train","train")]),
	"dataset/wiki19L-wor/.valid": ("wikitext-19L-wor valid", True, [("valid","valid")]),
	"dataset/wiki19L-wor/.test":  ("wikitext-19L-wor test",  True, [("test","test")]),

	# wikitext-2 modified splits
	"dataset/wiki2M/.test1":  ("wikitext-2 test1",  True, [("test1","test1")]),
	"dataset/wiki2M/.test2":  ("wikitext-2 test2",  True, [("test2","test2")]),
	"dataset/wiki2M/.valid1": ("wikitext-2 valid1", True, [("valid1","valid1")]),

	# wikitext-2 resampled (R1–R4)
	"dataset/wiki2Resample/.full1":  ("wikitext-2 Resampled 1 full",  True, [("train","trainR1"), ("valid","validR1"), ("test","testR1")]),
	"dataset/wiki2Resample/.test1":  ("wikitext-2 Resampled 1 test",  True, [("test","testR1")]),
	"dataset/wiki2Resample/.train1": ("wikitext-2 Resampled 1 train", True, [("train","trainR1")]),
	"dataset/wiki2Resample/.valid1": ("wikitext-2 Resampled 1 valid", True, [("valid","validR1")]),
	"dataset/wiki2Resample/.full2":  ("wikitext-2 Resampled 2 full",  True, [("train","trainR2"), ("valid","validR2"), ("test","testR2")]),
	"dataset/wiki2Resample/.test2":  ("wikitext-2 Resampled 2 test",  True, [("test","testR2")]),
	"dataset/wiki2Resample/.train2": ("wikitext-2 Resampled 2 train", True, [("train","trainR2")]),
	"dataset/wiki2Resample/.valid2": ("wikitext-2 Resampled 2 valid", True, [("valid","validR2")]),
	"dataset/wiki2Resample/.full3":  ("wikitext-2 Resampled 3 full",  True, [("train","trainR3"), ("valid","validR3"), ("test","testR3")]),
	"dataset/wiki2Resample/.test3":  ("wikitext-2 Resampled 3 test",  True, [("test","testR3")]),
	"dataset/wiki2Resample/.train3": ("wikitext-2 Resampled 3 train", True, [("train","trainR3")]),
	"dataset/wiki2Resample/.valid3": ("wikitext-2 Resampled 3 valid", True, [("valid","validR3")]),
	"dataset/wiki2Resample/.full4":  ("wikitext-2 Resampled 4 full",  True, [("train","trainR4"), ("valid","validR4"), ("test","testR4")]),
	"dataset/wiki2Resample/.test4":  ("wikitext-2 Resampled 4 test",  True, [("test","testR4")]),
	"dataset/wiki2Resample/.train4": ("wikitext-2 Resampled 4 train", True, [("train","trainR4")]),
	"dataset/wiki2Resample/.valid4": ("wikitext-2 Resampled 4 valid", True, [("valid","validR4")]),

	# wikitext-2 samples (1–4)
	"dataset/wiki2Samples/.full1":  ("wikitext-2 Samples 1 full",  True, [("train","train_1"), ("valid","valid_1"), ("test","test_1")]),
	"dataset/wiki2Samples/.test1":  ("wikitext-2 Samples 1 test",  True, [("test","test_1")]),
	"dataset/wiki2Samples/.train1": ("wikitext-2 Samples 1 train", True, [("train","train_1")]),
	"dataset/wiki2Samples/.valid1": ("wikitext-2 Samples 1 valid", True, [("valid","valid_1")]),
	"dataset/wiki2Samples/.full2":  ("wikitext-2 Samples 2 full",  True, [("train","train_2"), ("valid","valid_2"), ("test","test_2")]),
	"dataset/wiki2Samples/.test2":  ("wikitext-2 Samples 2 test",  True, [("test","test_2")]),
	"dataset/wiki2Samples/.train2": ("wikitext-2 Samples 2 train", True, [("train","train_2")]),
	"dataset/wiki2Samples/.valid2": ("wikitext-2 Samples 2 valid", True, [("valid","valid_2")]),
	"dataset/wiki2Samples/.full3":  ("wikitext-2 Samples 3 full",  True, [("train","train_3"), ("valid","valid_3"), ("test","test_3")]),
	"dataset/wiki2Samples/.test3":  ("wikitext-2 Samples 3 test",  True, [("test","test_3")]),
	"dataset/wiki2Samples/.train3": ("wikitext-2 Samples 3 train", True, [("train","train_3")]),
	"dataset/wiki2Samples/.valid3": ("wikitext-2 Samples 3 valid", True, [("valid","valid_3")]),
	"dataset/wiki2Samples/.full4":  ("wikitext-2 Samples 4 full",  True, [("train","train_4"), ("valid","valid_4"), ("test","test_4")]),
	"dataset/wiki2Samples/.test4":  ("wikitext-2 Samples 4 test",  True, [("test","test_4")]),
	"dataset/wiki2Samples/.train4": ("wikitext-2 Samples 4 train", True, [("train","train_4")]),
	"dataset/wiki2Samples/.valid4": ("wikitext-2 Samples 4 valid", True, [("valid","valid_4")]),

	# wikitext-2 Homogenous (H1, H2, CH1, CH2)
	"dataset/wiki2Homogenous/.fullH1":   ("wikitext-2 Homogenous 1 full",          True, [("train","trainH1"), ("valid","validH1"), ("test","testH1")]),
	"dataset/wiki2Homogenous/.testH1":   ("wikitext-2 Homogenous 1 test",          True, [("test","testH1")]),
	"dataset/wiki2Homogenous/.trainH1":  ("wikitext-2 Homogenous 1 train",         True, [("train","trainH1")]),
	"dataset/wiki2Homogenous/.validH1":  ("wikitext-2 Homogenous 1 valid",         True, [("valid","validH1")]),
	"dataset/wiki2Homogenous/.fullH2":   ("wikitext-2 Homogenous 2 full",          True, [("train","trainH2"), ("valid","validH2"), ("test","testH2")]),
	"dataset/wiki2Homogenous/.testH2":   ("wikitext-2 Homogenous 2 test",          True, [("test","testH2")]),
	"dataset/wiki2Homogenous/.trainH2":  ("wikitext-2 Homogenous 2 train",         True, [("train","trainH2")]),
	"dataset/wiki2Homogenous/.validH2":  ("wikitext-2 Homogenous 2 valid",         True, [("valid","validH2")]),
	"dataset/wiki2Homogenous/.fullCH1":  ("wikitext-2 Homogenous Cleaned 1 full",  True, [("train","trainCH1"), ("valid","validCH1"), ("test","testCH1")]),
	"dataset/wiki2Homogenous/.testCH1":  ("wikitext-2 Homogenous Cleaned 1 test",  True, [("test","testCH1")]),
	"dataset/wiki2Homogenous/.trainCH1": ("wikitext-2 Homogenous Cleaned 1 train", True, [("train","trainCH1")]),
	"dataset/wiki2Homogenous/.validCH1": ("wikitext-2 Homogenous Cleaned 1 valid", True, [("valid","validCH1")]),
	"dataset/wiki2Homogenous/.fullCH2":  ("wikitext-2 Homogenous Cleaned 2 full",  True, [("train","trainCH2"), ("valid","validCH2"), ("test","testCH2")]),
	"dataset/wiki2Homogenous/.testCH2":  ("wikitext-2 Homogenous Cleaned 2 test",  True, [("test","testCH2")]),
	"dataset/wiki2Homogenous/.trainCH2": ("wikitext-2 Homogenous Cleaned 2 train", True, [("train","trainCH2")]),
	"dataset/wiki2Homogenous/.validCH2": ("wikitext-2 Homogenous Cleaned 2 valid", True, [("valid","validCH2")]),

	# Hutter Prize / text8 (direct file paths, no subdirectory)
	"dataset/hutter/enwik8":             ("Hutter Enwik8",                    False, [("enwik8", None)]),
	"dataset/hutter/text8":              ("Hutter Text8",                     False, [("text8",  None)]),
	"dataset/hutter/text8-small":        ("text8 small",                      False, [("train",  None)]),
	"dataset/hutter/text8-wo-r-2":       ("text8 without rare words 2",       False, [("full",   None)]),
	"dataset/hutter/text8-small-wo-r-2": ("text8 small without rare words 2", False, [("full",   None)]),
	"dataset/hutter/text8-small-wo-r-4": ("text8 small without rare words 4", False, [("full",   None)]),

	# text8w (subdirectory)
	"dataset/hutter/text8w/.full":  ("text8w full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/hutter/text8w/.train": ("text8w train", True, [("train","train")]),
	"dataset/hutter/text8w/.valid": ("text8w valid", True, [("valid","valid")]),
	"dataset/hutter/text8w/.test":  ("text8w test",  True, [("test","test")]),

	# text8w_S (subdirectory)
	"dataset/hutter/text8w_S/.full":  ("text8w_S full",  True, [("train","train"), ("valid","valid"), ("test","test")]),
	"dataset/hutter/text8w_S/.train": ("text8w_S train", True, [("train","train")]),
	"dataset/hutter/text8w_S/.valid": ("text8w_S valid", True, [("valid","valid")]),
	"dataset/hutter/text8w_S/.test":  ("text8w_S test",  True, [("test","test")]),

	# text8w_small (subdirectory, .txt extensions)
	"dataset/hutter/text8w_small/.full":  ("text8w small full",  True, [("train","train.txt"), ("valid","valid.txt"), ("test","test.txt")]),
	"dataset/hutter/text8w_small/.train": ("text8w small train", True, [("train","train.txt")]),
	"dataset/hutter/text8w_small/.valid": ("text8w small valid", True, [("valid","valid.txt")]),
	"dataset/hutter/text8w_small/.test":  ("text8w small test",  True, [("test","test.txt")]),

	# ptb_text8 (subdirectory, .txt extensions)
	"dataset/ptb_text8/.full":       ("ptb_text8 full",        True,  [("train","train.txt"), ("valid","valid.txt"), ("test","test.txt")]),
	"dataset/ptb_text8/.train":      ("ptb_text8 train",       True,  [("train","train.txt")]),
	"dataset/ptb_text8/.valid":      ("ptb_text8 valid",       True,  [("valid","valid.txt")]),
	"dataset/ptb_text8/.test":       ("ptb_text8 test",        True,  [("test","test.txt")]),
	"dataset/ptb_text8/train_small": ("ptb_text8 train small", False, [("train", None)]),
}


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
		self.dataArray.extend(self.__wordLine)
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
		if path not in _REGISTRY:
			print("Please check the dataset path supplied. No such path found: " + path)
			print("Valid paths are listed in the _REGISTRY dict in data.py")
			sys.exit(0)

		label, is_subdir, splits = _REGISTRY[path]
		print(label)

		if is_subdir:
			base = path.rsplit('.', 1)[0]
			for attr, filename in splits:
				setattr(self, attr, self.tokenize_file(os.path.join(base, filename)))
		else:
			for attr, _ in splits:
				setattr(self, attr, self.tokenize_file(path))

		return 1

	def tokenize_file(self, path):
		if not os.path.exists(path):
			raise FileNotFoundError("Dataset file not found: " + path)
		print(path)
		with open(path, 'r', encoding='utf-8') as f:
			for line in f:
				if self.ifwords == 1:
					words = line.split() + ['<eos>']
				else:
					# Replace <unk> with a sentinel that cannot appear as a real character
					words = list(line.strip().replace("<unk>", "\x00")) + [" "]

				for word in words:
					wordID = self.dictionary.add_word(word)
					self.sequentialData.add_to_list(wordID)
		self.sequentialData.add_data()
		print("Size of Vocabulary: {0}".format(self.dictionary.totalUnique))
