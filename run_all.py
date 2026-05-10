#!/usr/bin/env python

# System libs
import os
import argparse

# Installed libs
import numpy as np

import data
import mutual_information as mi


def main():
	parser = argparse.ArgumentParser(description='Complex system properties of LM datasets')

	parser.add_argument('--data', type=str, required=True, help='location of the data corpus (e.g. dataset/wiki2/.full)')
	parser.add_argument('--words', type=int, default=0, help="Tokenize strings on words or characters: 1 = Words, 0 = Characters")
	parser.add_argument('--cutoff', type=int, default=1000, help="Value of maximum D you need.")
	parser.add_argument('--mi_method', type=str, default="grassberger", help="MI calculation method, Choose standard = Standard Calculation, grassberger = Grassberger adjustments")
	parser.add_argument('--log_type', type=int, default=1, help="Choose Log Type, 0 = Log to the base e, 1 = log to the base 2, 2 = log to the base 10")
	parser.add_argument('--threads', type=int, default=1, help='Number of threads to spawn')
	parser.add_argument('--overlap', type=int, default=1, help="Allow overlaps between two independent substrings. 0 = No, 1 = Yes")
	parser.add_argument('--granularity', type=int, default=1, help='How big is the spacing')
	parser.add_argument('--clear', type=int, default=0, help="Clear old data file")
	parser.add_argument('--save_path', type=str, default="save_data.dat", help="Save the data")

	args = parser.parse_args()

	###############################################################################
	# Load data
	###############################################################################

	# Create output directories so the script is self-contained
	for d in ["experiments/datasetInIDs", "experiments/corpus", "experiments/zipf",
	          "experiments/ldds", "experiments/heaps", "experiments/taylors",
	          "experiments/ebelings", "experiments/recurrence"]:
		os.makedirs(d, exist_ok=True)

	corpus = data.Corpus(args.data, args.words)
	np.savetxt("experiments/datasetInIDs/"+args.save_path+".out",
	           np.asarray(corpus.sequentialData.dataArray, dtype=np.uint32),
	           delimiter=',', fmt="%d")

	print("Corpus loaded")

	###############################################################################
	# Zipf's Law
	###############################################################################

	IDs = [i[0] for i in corpus.dictionary.counter.most_common()]
	frequency = [i[1] for i in corpus.dictionary.counter.most_common()]

	np.savez('experiments/zipf/'+args.save_path, np.asarray(IDs), np.asarray(frequency))

	print("Zipf's Computed")

	###############################################################################
	# Long-range correlation
	###############################################################################

	save_path = 'experiments/ldds/'+args.save_path+".dat"

	if args.clear == 1:
		try:
			os.remove(save_path)
		except OSError:
			print(save_path+" file does not exist.")
	ldd = mi.MutualInformation(corpus, args.log_type, args.threads, save_path, args.overlap, args.mi_method, args.cutoff)

	print("LDDs Computed")

	###############################################################################
	# Heap's Law, Ebeling's Method, Taylor's Law
	###############################################################################

	#subprocess.run(["./main", args.save_path], capture_output=True)

	#print("Heaps, Ebelings, Taylor Computed")

	###############################################################################
	# Recurrence
	###############################################################################

	# recur = recurrence.Recurrence(corpus, 10)

	# save_path = 'experiments/recurrence/'+args.save_path+".dat"

	# try:
	# 	recur_file = open(save_path, 'wb')
	# 	pickle.dump(recur.recurrenceList, recur_file)
	# 	recur_file.close()
	# except:
	#     print("Something went wrong")

	# print("Recurrence Computed")

if __name__ == '__main__':
		main()
