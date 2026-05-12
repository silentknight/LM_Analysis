#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static void print_bar(unsigned long long done, unsigned long long total, const char *label) {
	int width = 30;
	int filled = (total > 0) ? (int)((double)done / total * width) : width;
	int pct    = (total > 0) ? (int)((double)done / total * 100) : 100;
	printf("\r  %-22s [", label);
	for (int p = 0; p < width; p++) printf(p < filled ? "#" : ".");
	printf("] %3d%%", pct);
	fflush(stdout);
}

int main(int argc, char *argv[]) {
	FILE *fo, *fo1, *fi;
	char text[200];
	unsigned long long int i, j, k, index=0, seqLength=0, w_lower, w_upper, weight;
	unsigned long *wordIDSeq, *heaps, *subSeqLength, noOfSubSeq, *wordFreqList, vocabularySize=1;
	double tempSum, *mean, *variance, *sd;
	time_t old_time;
	char datasetFile[512];
	char heapsFile[512];
	char taylorsFile[512];
	char ebelingsFile[512];
	const char *exp_dir;

	if(argc < 2)
	{
		printf("Usage: scaling_laws <save_path> [experiments_dir]\n");
		exit(0);
	}
	exp_dir = (argc >= 3) ? argv[2] : "experiments";

	// ---------------------------------------------------------------------------------------
	// Read sequential data
	// ---------------------------------------------------------------------------------------
	snprintf (datasetFile, sizeof(datasetFile), "%s/datasetInIDs/%s.csv", exp_dir, argv[1]);
	printf("%s\n", datasetFile);
	fi = fopen(datasetFile, "r");
	if(fi == NULL)
	{
		printf("No such file\n");
		exit(0);
	}

	while(fscanf(fi, "%199s", text) == 1)
	{
		seqLength++;
	}

	wordIDSeq = (unsigned long*) malloc(seqLength*sizeof(unsigned long));
	rewind(fi);
	while(fscanf(fi, "%199s", text) == 1)
	{
		wordIDSeq[index] = strtoul(text, NULL, 10);
		if(wordIDSeq[index] > vocabularySize)
		{
			vocabularySize = wordIDSeq[index];
		}
		index++;
		if (index % 100000 == 0 || index == seqLength)
			print_bar(index, seqLength, "Loading corpus");
	}
	printf("\n");
	vocabularySize++;

	fclose(fi);
	printf("Corpus loaded  (%llu tokens, vocab %lu)\n", seqLength, vocabularySize);

	// ---------------------------------------------------------------------------------------
	// Heaps Law
	// ---------------------------------------------------------------------------------------
	wordFreqList = (unsigned long*) malloc(vocabularySize*sizeof(unsigned long));
	for(i=0; i<vocabularySize; i++)
	{
		wordFreqList[i]=0;
	}

	heaps = (unsigned long*) malloc(seqLength*sizeof(unsigned long));
	for(i=0; i<seqLength; i++)
	{
		heaps[i]=0;
	}

	for(i=0; i<seqLength; i++)
	{
		if(wordFreqList[wordIDSeq[i]] == 0)
		{
			wordFreqList[wordIDSeq[i]]++;
			heaps[i] = (i == 0) ? 1 : heaps[i-1] + 1;
		}
		else
		{
			heaps[i] = heaps[i-1];
		}
		if (i % 100000 == 0 || i == seqLength - 1)
			print_bar(i + 1, seqLength, "Heap's Law");
	}
	printf("\n");

	snprintf (heapsFile, sizeof(heapsFile), "%s/heaps/%s_heaps.csv", exp_dir, argv[1]);
	printf("%s\n", heapsFile);
	fo = fopen(heapsFile, "w");
	if(fo == NULL)
	{
		printf("No such file\n");
		exit(0);
	}

	for (i=0; i<seqLength; ++i)
	{
		fprintf(fo, "%lu,", heaps[i]);
	}

	fclose(fo);
	printf("Heaps computed\n");

	// ---------------------------------------------------------------------------------------
	// Taylor's Law	& Ebeling's Law
	// ---------------------------------------------------------------------------------------
	index=0;
	noOfSubSeq = (unsigned long)log10((double)seqLength);

	snprintf (taylorsFile, sizeof(taylorsFile), "%s/taylors/%s_taylors.csv", exp_dir, argv[1]);
	printf("%s\n", taylorsFile);
	fo = fopen(taylorsFile, "w");
	if(fo == NULL)
	{
		printf("Cannot open taylors file\n");
		exit(0);
	}

	snprintf (ebelingsFile, sizeof(ebelingsFile), "%s/ebelings/%s_ebelings.csv", exp_dir, argv[1]);
	printf("%s\n", ebelingsFile);
	fo1 = fopen(ebelingsFile, "w");
	if(fo1 == NULL)
	{
		printf("Cannot open ebelings file\n");
		exit(0);
	}

	variance = (double*) malloc(noOfSubSeq*sizeof(double));
	for(k=0; k<noOfSubSeq; k++)
	{
		variance[k]=0;
	}

	subSeqLength = (unsigned long*) malloc(noOfSubSeq*sizeof(unsigned long));
	for(k=0; k<noOfSubSeq; k++)
	{
		subSeqLength[k]=0;
	}

	for(i=10; i<seqLength; i=i*10)
	{
		char label[32];
		snprintf(label, sizeof(label), "Taylor i=%-9llu", i);
		fprintf(fo, "%llu\n", i);
		tempSum = 0;
		subSeqLength[index] = i;

		mean = (double*) malloc(vocabularySize*sizeof(double));
		for(k=0; k<vocabularySize; k++) mean[k] = 0;

		// Compute mean: O(seqLength) weighted single-pass.
		// Each corpus position k contributes to exactly
		//   min(seqLength-i, k) - max(0, k-i+1) + 1  windows,
		// so we accumulate that weight instead of iterating over every window.
		// The original nested loop was O(seqLength * i) which dominated runtime.
		for(k=0; k<seqLength; k++)
		{
			w_lower = (k >= i - 1) ? (k - i + 1) : 0;
			w_upper = (k <= seqLength - i) ? k : (seqLength - i);
			weight = w_upper - w_lower + 1;
			mean[wordIDSeq[k]] += (double)weight;
		}
		for(k=0; k<vocabularySize; k++)
		{
			mean[k] = mean[k]/(double)(seqLength-i+1);
			fprintf(fo, "%f,", mean[k]);
		}
		fprintf(fo, "\n");

		// Lazy variance accumulation — O(N+V) instead of O(N*V).
		// sq_contrib[k] = (current_count[k] - mean[k])^2, updated only when k's count
		// changes (at most 2 words change per window step).  sd[k] accumulates
		// sq_contrib[k] * run_length for each run between changes, then is finalised
		// at the end with one O(V) pass.
		sd = (double*) calloc(vocabularySize, sizeof(double));
		double *sq_contrib = (double*) calloc(vocabularySize, sizeof(double));
		unsigned long long *last_event = (unsigned long long*) calloc(vocabularySize, sizeof(unsigned long long));

		for(k=0; k<vocabularySize; k++) wordFreqList[k] = 0;

		unsigned long long int total_j = seqLength - i + 1;
		unsigned long long int report_step = total_j / 40;
		if (report_step < 1) report_step = 1;

		// Initialise first window [0, i-1]
		for(k=0; k<i; k++)
			wordFreqList[wordIDSeq[k]]++;

		// Seed sq_contrib from first window — O(V) one-time cost
		for(k=0; k<vocabularySize; k++) {
			double dev = (double)wordFreqList[k] - mean[k];
			sq_contrib[k] = dev * dev;
		}
		// last_event[k] = 0 for all k (calloc)
		print_bar(1, total_j, label);

		// Slide window: only 2 words change per step — O(1) work per step
		for(j=1; j<total_j; j++)
		{
			unsigned long out_word = wordIDSeq[j-1];
			unsigned long in_word  = wordIDSeq[j+i-1];

			if (out_word != in_word) {
				// Flush accumulated contribution for changed words
				sd[out_word] += sq_contrib[out_word] * (double)(j - last_event[out_word]);
				sd[in_word]  += sq_contrib[in_word]  * (double)(j - last_event[in_word]);

				wordFreqList[out_word]--;
				wordFreqList[in_word]++;

				double dev_out = (double)wordFreqList[out_word] - mean[out_word];
				double dev_in  = (double)wordFreqList[in_word]  - mean[in_word];
				sq_contrib[out_word] = dev_out * dev_out;
				sq_contrib[in_word]  = dev_in  * dev_in;
				last_event[out_word] = j;
				last_event[in_word]  = j;
			}
			// else: same word exits and enters — net count unchanged, lazy state is still valid

			if (j % report_step == 0 || j == total_j - 1)
				print_bar(j + 1, total_j, label);
		}
		printf("\n");

		// Finalise: flush remaining contributions and compute standard deviation — O(V)
		for (k=0; k<vocabularySize; k++)
		{
			sd[k] += sq_contrib[k] * (double)(total_j - last_event[k]);
			sd[k] = sd[k] / (double)(seqLength-i+1);
			tempSum += sd[k];
			sd[k] = sqrt(sd[k]);
			fprintf(fo, "%f,", sd[k]);
		}
		fprintf(fo, "\n");
		fprintf(fo1, "%llu:%f,", i, tempSum);

		free(sq_contrib);
		free(last_event);

		variance[index++] = tempSum;
		free(mean);
		free(sd);
	}
	fclose(fo);
	fclose(fo1);

	free(wordIDSeq);
	free(heaps);
	free(wordFreqList);
	free(variance);
	free(subSeqLength);

	printf("Taylor's Law & Ebeling's Method computed\n");
}
