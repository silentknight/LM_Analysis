#!/usr/bin/env python

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np

import data
from analysis import mi


def main():
	parser = argparse.ArgumentParser(description='Complex system properties of LM datasets')

	parser.add_argument('--data', type=str, required=True, help='location of the data corpus (e.g. dataset/wiki2/.full)')
	parser.add_argument('--words', type=int, default=0, help="Tokenize strings on words or characters: 1 = Words, 0 = Characters")
	parser.add_argument('--cutoff', type=int, default=1000, help="Value of maximum D you need.")
	parser.add_argument('--mi_method', type=str, default="grassberger", help="MI calculation method: standard or grassberger")
	parser.add_argument('--log_type', type=int, default=1, help="Log base: 0=e, 1=2, 2=10")
	parser.add_argument('--threads', type=int, default=1, help='Number of threads to spawn')
	parser.add_argument('--overlap', type=int, default=1, help="Allow overlaps between two independent substrings. 0=No, 1=Yes")
	parser.add_argument('--granularity', type=int, default=1, help='How big is the spacing')
	parser.add_argument('--clear', type=int, default=0, help="Clear old LDD data file: 0=No, 1=Yes")
	parser.add_argument('--save_path', type=str, default="save_data", help="Base name for all output files")
	parser.add_argument('--processes', nargs='+',
	                    default=['mi', 'pmi', 'heaps', 'recurrence'],
	                    choices=['mi', 'pmi', 'heaps', 'recurrence'],
	                    help='Analyses to run (default: all). e.g. --processes mi pmi heaps')
	parser.add_argument('--pmi_dir', type=str, default=None,
	                    help='Output directory for PMI matrices (default: experiments/pmi/<save_path>)')

	args = parser.parse_args()

	from data.loader import _REGISTRY
	if args.data not in _REGISTRY:
		parser.error(f"Unknown data path '{args.data}'. Valid paths are listed in src/data/loader.py")

	for d in ["experiments/datasetInIDs", "experiments/corpus", "experiments/zipf",
	          "experiments/ldds", "experiments/heaps", "experiments/taylors",
	          "experiments/ebelings", "experiments/recurrence", "experiments/pmi"]:
		os.makedirs(d, exist_ok=True)

	###############################################################################
	# Load data (always required as input to all analyses)
	###############################################################################

	corpus = data.Corpus(args.data, args.words)
	np.savetxt("experiments/datasetInIDs/" + args.save_path + ".csv",
	           np.asarray(corpus.sequentialData.dataArray, dtype=np.uint32),
	           fmt="%d")
	print("Corpus loaded")

	###############################################################################
	# Zipf's Law (always computed — negligible cost)
	###############################################################################

	ranked = corpus.dictionary.counter.most_common()
	IDs, frequency = zip(*ranked) if ranked else ([], [])
	np.savez_compressed('experiments/zipf/' + args.save_path,
	                    ids=np.asarray(IDs), frequency=np.asarray(frequency))
	print("Zipf's Computed")

	###############################################################################
	# Mutual Information / LDD
	###############################################################################

	if 'mi' in args.processes:
		ldd_path = 'experiments/ldds/' + args.save_path + '.csv'
		if args.clear == 1:
			try:
				os.remove(ldd_path)
			except OSError:
				print(ldd_path + " file does not exist.")
		mi.MutualInformation(corpus, args.log_type, args.threads, ldd_path,
		                     args.overlap, args.mi_method, args.cutoff)
		print("LDDs Computed")

	###############################################################################
	# Pointwise Mutual Information
	###############################################################################

	if 'pmi' in args.processes:
		from analysis import pmi as pmi_mod
		pmi_dir = args.pmi_dir or os.path.join("experiments/pmi", args.save_path)
		os.makedirs(pmi_dir, exist_ok=True)
		pmi_mod.PointwiseMutualInformation(corpus, args.log_type, args.threads,
		                                   pmi_dir, args.overlap, "standard", args.cutoff)
		print("PMI Computed")

	###############################################################################
	# Heap's Law, Taylor's Law, Ebeling's Method  (C binary)
	###############################################################################

	if 'heaps' in args.processes:
		binary = 'src/analysis/scaling_laws'
		if not os.path.exists(binary):
			print("Compiling src/analysis/scaling_laws.c ...")
			result = subprocess.run(['gcc', '-O2', '-o', binary, 'src/analysis/scaling_laws.c', '-lm'],
			                        capture_output=True, text=True)
			if result.returncode != 0:
				print("Compilation failed:\n" + result.stderr)
			else:
				print("Compiled src/analysis/scaling_laws.c")
		if os.path.exists(binary):
			subprocess.run([binary, args.save_path, 'experiments'])
			print("Heaps, Taylors, Ebelings Computed")

	###############################################################################
	# Recurrence
	###############################################################################

	if 'recurrence' in args.processes:
		from analysis import recurrence as rec_mod
		recur = rec_mod.Recurrence(corpus, args.threads)
		rec_path = 'experiments/recurrence/' + args.save_path + '.npz'
		np.savez_compressed(rec_path, **{str(k): v for k, v in recur.recurrenceList.items()})
		print("Recurrence Computed")



if __name__ == '__main__':
	main()
