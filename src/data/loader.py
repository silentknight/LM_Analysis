#!/usr/bin/env python

import os
import sys
from collections import Counter

# Registry: path_key → (label, data_type, language, is_subdir, splits)
#
# data_type : broad modality — "text", "code", "image", …
# language  : natural language ("english", "french", …) or programming language
#             ("python", "c", …) or other descriptor ("mixed", "bytes", …)
# is_subdir : True  → base = key.rsplit('.', 1)[0]; files resolved relative to base
#             False → file is the key itself
# fmt: off
_REGISTRY = {
    # -------------------------------------------------------------------------
    # Natural language — English
    # -------------------------------------------------------------------------

    # Penn Treebank
    "dataset/text/english/ptb/.full":  ("Penn Tree Bank Full",  "text", "english", True,  [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/ptb/.train": ("Penn Tree Bank Train", "text", "english", True,  [("train", "train")]),
    "dataset/text/english/ptb/.valid": ("Penn Tree Bank Valid", "text", "english", True,  [("valid", "valid")]),
    "dataset/text/english/ptb/.test":  ("Penn Tree Bank Test",  "text", "english", True,  [("test",  "test")]),

    # wikitext-2
    "dataset/text/english/wiki2/.full":  ("wikitext-2 full",  "text", "english", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/wiki2/.train": ("wikitext-2 train", "text", "english", True, [("train", "train")]),
    "dataset/text/english/wiki2/.valid": ("wikitext-2 valid", "text", "english", True, [("valid", "valid")]),
    "dataset/text/english/wiki2/.test":  ("wikitext-2 test",  "text", "english", True, [("test",  "test")]),

    # wikitext-2 raw
    "dataset/text/english/wiki2R/.full":  ("wikitext-2 raw full",  "text", "english", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/wiki2R/.train": ("wikitext-2 raw train", "text", "english", True, [("train", "train")]),
    "dataset/text/english/wiki2R/.valid": ("wikitext-2 raw valid", "text", "english", True, [("valid", "valid")]),
    "dataset/text/english/wiki2R/.test":  ("wikitext-2 raw test",  "text", "english", True, [("test",  "test")]),

    # wikitext-2 cleaned
    "dataset/text/english/wiki2C/.full":   ("wikitext-2 cleaned full",   "text", "english", True, [("train", "trainC"), ("valid", "validC"), ("test", "testC")]),
    "dataset/text/english/wiki2C/.train":  ("wikitext-2 cleaned train",  "text", "english", True, [("trainC",  "trainC")]),
    "dataset/text/english/wiki2C/.valid":  ("wikitext-2 cleaned valid",  "text", "english", True, [("validC",  "validC")]),
    "dataset/text/english/wiki2C/.valid1": ("wikitext-2 cleaned valid1", "text", "english", True, [("validC1", "validC1")]),
    "dataset/text/english/wiki2C/.test":   ("wikitext-2 cleaned test",   "text", "english", True, [("testC",   "testC")]),
    "dataset/text/english/wiki2C/.test1":  ("wikitext-2 cleaned test1",  "text", "english", True, [("testC1",  "testC1")]),
    "dataset/text/english/wiki2C/.test2":  ("wikitext-2 cleaned test2",  "text", "english", True, [("testC2",  "testC2")]),

    # wikitext-2 modified splits
    "dataset/text/english/wiki2M/.test1":  ("wikitext-2 test1",  "text", "english", True, [("test1",  "test1")]),
    "dataset/text/english/wiki2M/.test2":  ("wikitext-2 test2",  "text", "english", True, [("test2",  "test2")]),
    "dataset/text/english/wiki2M/.valid1": ("wikitext-2 valid1", "text", "english", True, [("valid1", "valid1")]),

    # wikitext-2 resampled (R1–R4)
    "dataset/text/english/wiki2Resample/.full1":  ("wikitext-2 Resampled 1 full",  "text", "english", True, [("train", "trainR1"), ("valid", "validR1"), ("test", "testR1")]),
    "dataset/text/english/wiki2Resample/.train1": ("wikitext-2 Resampled 1 train", "text", "english", True, [("train", "trainR1")]),
    "dataset/text/english/wiki2Resample/.valid1": ("wikitext-2 Resampled 1 valid", "text", "english", True, [("valid", "validR1")]),
    "dataset/text/english/wiki2Resample/.test1":  ("wikitext-2 Resampled 1 test",  "text", "english", True, [("test",  "testR1")]),
    "dataset/text/english/wiki2Resample/.full2":  ("wikitext-2 Resampled 2 full",  "text", "english", True, [("train", "trainR2"), ("valid", "validR2"), ("test", "testR2")]),
    "dataset/text/english/wiki2Resample/.train2": ("wikitext-2 Resampled 2 train", "text", "english", True, [("train", "trainR2")]),
    "dataset/text/english/wiki2Resample/.valid2": ("wikitext-2 Resampled 2 valid", "text", "english", True, [("valid", "validR2")]),
    "dataset/text/english/wiki2Resample/.test2":  ("wikitext-2 Resampled 2 test",  "text", "english", True, [("test",  "testR2")]),
    "dataset/text/english/wiki2Resample/.full3":  ("wikitext-2 Resampled 3 full",  "text", "english", True, [("train", "trainR3"), ("valid", "validR3"), ("test", "testR3")]),
    "dataset/text/english/wiki2Resample/.train3": ("wikitext-2 Resampled 3 train", "text", "english", True, [("train", "trainR3")]),
    "dataset/text/english/wiki2Resample/.valid3": ("wikitext-2 Resampled 3 valid", "text", "english", True, [("valid", "validR3")]),
    "dataset/text/english/wiki2Resample/.test3":  ("wikitext-2 Resampled 3 test",  "text", "english", True, [("test",  "testR3")]),
    "dataset/text/english/wiki2Resample/.full4":  ("wikitext-2 Resampled 4 full",  "text", "english", True, [("train", "trainR4"), ("valid", "validR4"), ("test", "testR4")]),
    "dataset/text/english/wiki2Resample/.train4": ("wikitext-2 Resampled 4 train", "text", "english", True, [("train", "trainR4")]),
    "dataset/text/english/wiki2Resample/.valid4": ("wikitext-2 Resampled 4 valid", "text", "english", True, [("valid", "validR4")]),
    "dataset/text/english/wiki2Resample/.test4":  ("wikitext-2 Resampled 4 test",  "text", "english", True, [("test",  "testR4")]),

    # wikitext-2 samples (1–4)
    "dataset/text/english/wiki2Samples/.full1":  ("wikitext-2 Samples 1 full",  "text", "english", True, [("train", "train_1"), ("valid", "valid_1"), ("test", "test_1")]),
    "dataset/text/english/wiki2Samples/.train1": ("wikitext-2 Samples 1 train", "text", "english", True, [("train", "train_1")]),
    "dataset/text/english/wiki2Samples/.valid1": ("wikitext-2 Samples 1 valid", "text", "english", True, [("valid", "valid_1")]),
    "dataset/text/english/wiki2Samples/.test1":  ("wikitext-2 Samples 1 test",  "text", "english", True, [("test",  "test_1")]),
    "dataset/text/english/wiki2Samples/.full2":  ("wikitext-2 Samples 2 full",  "text", "english", True, [("train", "train_2"), ("valid", "valid_2"), ("test", "test_2")]),
    "dataset/text/english/wiki2Samples/.train2": ("wikitext-2 Samples 2 train", "text", "english", True, [("train", "train_2")]),
    "dataset/text/english/wiki2Samples/.valid2": ("wikitext-2 Samples 2 valid", "text", "english", True, [("valid", "valid_2")]),
    "dataset/text/english/wiki2Samples/.test2":  ("wikitext-2 Samples 2 test",  "text", "english", True, [("test",  "test_2")]),
    "dataset/text/english/wiki2Samples/.full3":  ("wikitext-2 Samples 3 full",  "text", "english", True, [("train", "train_3"), ("valid", "valid_3"), ("test", "test_3")]),
    "dataset/text/english/wiki2Samples/.train3": ("wikitext-2 Samples 3 train", "text", "english", True, [("train", "train_3")]),
    "dataset/text/english/wiki2Samples/.valid3": ("wikitext-2 Samples 3 valid", "text", "english", True, [("valid", "valid_3")]),
    "dataset/text/english/wiki2Samples/.test3":  ("wikitext-2 Samples 3 test",  "text", "english", True, [("test",  "test_3")]),
    "dataset/text/english/wiki2Samples/.full4":  ("wikitext-2 Samples 4 full",  "text", "english", True, [("train", "train_4"), ("valid", "valid_4"), ("test", "test_4")]),
    "dataset/text/english/wiki2Samples/.train4": ("wikitext-2 Samples 4 train", "text", "english", True, [("train", "train_4")]),
    "dataset/text/english/wiki2Samples/.valid4": ("wikitext-2 Samples 4 valid", "text", "english", True, [("valid", "valid_4")]),
    "dataset/text/english/wiki2Samples/.test4":  ("wikitext-2 Samples 4 test",  "text", "english", True, [("test",  "test_4")]),

    # wikitext-2 Homogenous (H1, H2, CH1, CH2)
    "dataset/text/english/wiki2Homogenous/.fullH1":  ("wikitext-2 Homogenous 1 full",          "text", "english", True, [("train", "trainH1"),  ("valid", "validH1"),  ("test", "testH1")]),
    "dataset/text/english/wiki2Homogenous/.trainH1": ("wikitext-2 Homogenous 1 train",         "text", "english", True, [("train", "trainH1")]),
    "dataset/text/english/wiki2Homogenous/.validH1": ("wikitext-2 Homogenous 1 valid",         "text", "english", True, [("valid", "validH1")]),
    "dataset/text/english/wiki2Homogenous/.testH1":  ("wikitext-2 Homogenous 1 test",          "text", "english", True, [("test",  "testH1")]),
    "dataset/text/english/wiki2Homogenous/.fullH2":  ("wikitext-2 Homogenous 2 full",          "text", "english", True, [("train", "trainH2"),  ("valid", "validH2"),  ("test", "testH2")]),
    "dataset/text/english/wiki2Homogenous/.trainH2": ("wikitext-2 Homogenous 2 train",         "text", "english", True, [("train", "trainH2")]),
    "dataset/text/english/wiki2Homogenous/.validH2": ("wikitext-2 Homogenous 2 valid",         "text", "english", True, [("valid", "validH2")]),
    "dataset/text/english/wiki2Homogenous/.testH2":  ("wikitext-2 Homogenous 2 test",          "text", "english", True, [("test",  "testH2")]),
    "dataset/text/english/wiki2Homogenous/.fullCH1": ("wikitext-2 Homogenous Cleaned 1 full",  "text", "english", True, [("train", "trainCH1"), ("valid", "validCH1"), ("test", "testCH1")]),
    "dataset/text/english/wiki2Homogenous/.trainCH1":("wikitext-2 Homogenous Cleaned 1 train", "text", "english", True, [("train", "trainCH1")]),
    "dataset/text/english/wiki2Homogenous/.validCH1":("wikitext-2 Homogenous Cleaned 1 valid", "text", "english", True, [("valid", "validCH1")]),
    "dataset/text/english/wiki2Homogenous/.testCH1": ("wikitext-2 Homogenous Cleaned 1 test",  "text", "english", True, [("test",  "testCH1")]),
    "dataset/text/english/wiki2Homogenous/.fullCH2": ("wikitext-2 Homogenous Cleaned 2 full",  "text", "english", True, [("train", "trainCH2"), ("valid", "validCH2"), ("test", "testCH2")]),
    "dataset/text/english/wiki2Homogenous/.trainCH2":("wikitext-2 Homogenous Cleaned 2 train", "text", "english", True, [("train", "trainCH2")]),
    "dataset/text/english/wiki2Homogenous/.validCH2":("wikitext-2 Homogenous Cleaned 2 valid", "text", "english", True, [("valid", "validCH2")]),
    "dataset/text/english/wiki2Homogenous/.testCH2": ("wikitext-2 Homogenous Cleaned 2 test",  "text", "english", True, [("test",  "testCH2")]),

    # wikitext-103
    "dataset/text/english/wiki103/.full":   ("wikitext-103 full",          "text", "english", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/wiki103/.train":  ("wikitext-103 train",         "text", "english", True, [("train", "train")]),
    "dataset/text/english/wiki103/.valid":  ("wikitext-103 valid",         "text", "english", True, [("valid", "valid")]),
    "dataset/text/english/wiki103/.test":   ("wikitext-103 test",          "text", "english", True, [("test",  "test")]),
    "dataset/text/english/wiki103C/.full":  ("wikitext-103 cleaned full",  "text", "english", True, [("train", "trainC"), ("valid", "validC"), ("test", "testC")]),
    "dataset/text/english/wiki103C/.train": ("wikitext-103 cleaned train", "text", "english", True, [("trainC", "trainC")]),
    "dataset/text/english/wiki103C/.valid": ("wikitext-103 cleaned valid", "text", "english", True, [("validC", "validC")]),
    "dataset/text/english/wiki103C/.test":  ("wikitext-103 cleaned test",  "text", "english", True, [("testC",  "testC")]),
    "dataset/text/english/wiki103R/.full":  ("wikitext-103 raw full",      "text", "english", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/wiki103R/.train": ("wikitext-103 raw train",     "text", "english", True, [("train", "train")]),
    "dataset/text/english/wiki103R/.valid": ("wikitext-103 raw valid",     "text", "english", True, [("valid", "valid")]),
    "dataset/text/english/wiki103R/.test":  ("wikitext-103 raw test",      "text", "english", True, [("test",  "test")]),

    # wikitext-19
    "dataset/text/english/wiki19/.full":      ("wikitext-19 full",                  "text", "english", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/wiki19/.train":     ("wikitext-19 train",                 "text", "english", True, [("train", "train")]),
    "dataset/text/english/wiki19/.valid":     ("wikitext-19 valid",                 "text", "english", True, [("valid", "valid")]),
    "dataset/text/english/wiki19/.test":      ("wikitext-19 test",                  "text", "english", True, [("test",  "test")]),
    "dataset/text/english/wiki19C/.full":     ("wikitext-19 cleaned full",          "text", "english", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/wiki19C/.train":    ("wikitext-19 cleaned train",         "text", "english", True, [("train", "train")]),
    "dataset/text/english/wiki19C/.valid":    ("wikitext-19 cleaned valid",         "text", "english", True, [("valid", "valid")]),
    "dataset/text/english/wiki19C/.test":     ("wikitext-19 cleaned test",          "text", "english", True, [("test",  "test")]),
    "dataset/text/english/wiki19L/.full":     ("wikitext-19L Text8-Like full",      "text", "english", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/wiki19L/.train":    ("wikitext-19L Text8-Like train",     "text", "english", True, [("train", "train")]),
    "dataset/text/english/wiki19L/.valid":    ("wikitext-19L Text8-Like valid",     "text", "english", True, [("valid", "valid")]),
    "dataset/text/english/wiki19L/.test":     ("wikitext-19L Text8-Like test",      "text", "english", True, [("test",  "test")]),
    "dataset/text/english/wiki19L-wor/.full": ("wikitext-19L-wor full",             "text", "english", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/wiki19L-wor/.train":("wikitext-19L-wor train",            "text", "english", True, [("train", "train")]),
    "dataset/text/english/wiki19L-wor/.valid":("wikitext-19L-wor valid",            "text", "english", True, [("valid", "valid")]),
    "dataset/text/english/wiki19L-wor/.test": ("wikitext-19L-wor test",             "text", "english", True, [("test",  "test")]),

    # OpenWebText
    "dataset/text/english/openweb/.full":  ("OpenWebText full",  "text", "english", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/openweb/.train": ("OpenWebText train", "text", "english", True, [("train", "train")]),
    "dataset/text/english/openweb/.valid": ("OpenWebText valid", "text", "english", True, [("valid", "valid")]),
    "dataset/text/english/openweb/.test":  ("OpenWebText test",  "text", "english", True, [("test",  "test")]),

    # English Wikipedia
    "dataset/text/english/wikipedia/.full":  ("Wikipedia full",  "text", "english", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/english/wikipedia/.train": ("Wikipedia train", "text", "english", True, [("train", "train")]),
    "dataset/text/english/wikipedia/.valid": ("Wikipedia valid", "text", "english", True, [("valid", "valid")]),
    "dataset/text/english/wikipedia/.test":  ("Wikipedia test",  "text", "english", True, [("test",  "test")]),

    # -------------------------------------------------------------------------
    # Natural language — bytes / mixed  (Hutter Prize)
    # -------------------------------------------------------------------------

    "dataset/text/bytes/hutter/enwik8":            ("Hutter Enwik8",                    "text", "bytes", False, [("enwik8", None)]),
    "dataset/text/bytes/hutter/text8":             ("Hutter Text8",                     "text", "bytes", False, [("text8",  None)]),
    "dataset/text/bytes/hutter/text8-small":       ("text8 small",                      "text", "bytes", False, [("train",  None)]),
    "dataset/text/bytes/hutter/text8-wo-r-2":      ("text8 without rare words 2",       "text", "bytes", False, [("full",   None)]),
    "dataset/text/bytes/hutter/text8-small-wo-r-2":("text8 small without rare words 2", "text", "bytes", False, [("full",   None)]),
    "dataset/text/bytes/hutter/text8-small-wo-r-4":("text8 small without rare words 4", "text", "bytes", False, [("full",   None)]),

    "dataset/text/bytes/hutter/text8w/.full":  ("text8w full",  "text", "bytes", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/bytes/hutter/text8w/.train": ("text8w train", "text", "bytes", True, [("train", "train")]),
    "dataset/text/bytes/hutter/text8w/.valid": ("text8w valid", "text", "bytes", True, [("valid", "valid")]),
    "dataset/text/bytes/hutter/text8w/.test":  ("text8w test",  "text", "bytes", True, [("test",  "test")]),

    "dataset/text/bytes/hutter/text8w_S/.full":  ("text8w_S full",  "text", "bytes", True, [("train", "train"), ("valid", "valid"), ("test", "test")]),
    "dataset/text/bytes/hutter/text8w_S/.train": ("text8w_S train", "text", "bytes", True, [("train", "train")]),
    "dataset/text/bytes/hutter/text8w_S/.valid": ("text8w_S valid", "text", "bytes", True, [("valid", "valid")]),
    "dataset/text/bytes/hutter/text8w_S/.test":  ("text8w_S test",  "text", "bytes", True, [("test",  "test")]),

    "dataset/text/bytes/hutter/text8w_small/.full":  ("text8w small full",  "text", "bytes", True, [("train", "train.txt"), ("valid", "valid.txt"), ("test", "test.txt")]),
    "dataset/text/bytes/hutter/text8w_small/.train": ("text8w small train", "text", "bytes", True, [("train", "train.txt")]),
    "dataset/text/bytes/hutter/text8w_small/.valid": ("text8w small valid", "text", "bytes", True, [("valid", "valid.txt")]),
    "dataset/text/bytes/hutter/text8w_small/.test":  ("text8w small test",  "text", "bytes", True, [("test",  "test.txt")]),

    "dataset/text/bytes/ptb_text8/.full":       ("ptb_text8 full",        "text", "bytes", True,  [("train", "train.txt"), ("valid", "valid.txt"), ("test", "test.txt")]),
    "dataset/text/bytes/ptb_text8/.train":      ("ptb_text8 train",       "text", "bytes", True,  [("train", "train.txt")]),
    "dataset/text/bytes/ptb_text8/.valid":      ("ptb_text8 valid",       "text", "bytes", True,  [("valid", "valid.txt")]),
    "dataset/text/bytes/ptb_text8/.test":       ("ptb_text8 test",        "text", "bytes", True,  [("test",  "test.txt")]),
    "dataset/text/bytes/ptb_text8/train_small": ("ptb_text8 train small", "text", "bytes", False, [("train", None)]),
}
# fmt: on


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
        self.averageLength = (
            self.averageLength * len(self.wordCountList) + len(self.__wordLine)
        ) / (len(self.wordCountList) + 1)
        self.wordCountList.append(len(self.__wordLine))
        self.fileStartIndex.append(self.totalLength)
        self.totalLength += len(self.__wordLine)
        self.__wordLine = []


class Corpus(object):
    def __init__(self, path, tokenizer):
        self.dictionary = Dictionary()
        self.sequentialData = SequentialData()
        self.datainfo = path
        self.tokenizer = tokenizer
        self.completion = self.choose_dataset(self.datainfo)
        print("Total Length: {0}".format(self.sequentialData.totalLength))

    def choose_dataset(self, path):
        if path not in _REGISTRY:
            print("Please check the dataset path supplied. No such path found: " + path)
            print("Valid paths are listed in the _REGISTRY dict in data.py")
            sys.exit(0)

        label, data_type, language, is_subdir, splits = _REGISTRY[path]
        self.data_type = data_type
        self.language = language
        print(label)

        if is_subdir:
            base = path.rsplit(".", 1)[0]
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
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                for word in self.tokenizer.tokenize(line):
                    wordID = self.dictionary.add_word(word)
                    self.sequentialData.add_to_list(wordID)
        self.sequentialData.add_data()
        print("Size of Vocabulary: {0}".format(self.dictionary.totalUnique))
