#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
	FILE *fo, *fo1, *fi;
	char text[200];
	unsigned long long int i, j, k, temp_k, index=0, seqLength=0;
	unsigned long *wordIDSeq, *heaps, *subSeqLength, noOfSubSeq, ret, *wordFreqList, vocabularySize=1;
	double tempSum, *mean, *variance, *sd;
	time_t old_time;
	char datasetFile[100];
	char heapsFile[100];
	char taylorsFile[100];
	char ebelingsFile[100];

	if(argc < 2)
	{
		printf("Please provide filenames for processing\n");
		exit(0);
	}

	// ---------------------------------------------------------------------------------------
	// Read sequential data
	// ---------------------------------------------------------------------------------------
	snprintf (datasetFile, 100, "experiments/datasetInIDs/%s.out", argv[1]);
	printf("%s\n", datasetFile);
	fi = fopen(datasetFile, "r");
	if(fi == NULL)
	{
		printf("No such file\n");
		exit(0);
	}

	while(!feof(fi))
	{
		fscanf(fi,"%s", text);
		seqLength++;
	}

	wordIDSeq = (unsigned long*) malloc(seqLength*sizeof(unsigned long));
	rewind(fi);
	while(!feof(fi))
	{
		fscanf(fi,"%s", text);
		wordIDSeq[index] = atoi(text);
		if(wordIDSeq[index] > vocabularySize)
		{
			vocabularySize = wordIDSeq[index];
		}
		index++;
	}
	vocabularySize++;

	fclose(fi);
	printf("Corpus loaded\n");

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

			if(i==0)
			{

				heaps[i]=1;
			}
			else
			{
				heaps[i] = heaps[i-1]+1;
			}
		}
		else
		{
			heaps[i] = heaps[i-1];
		}
	}

	snprintf (heapsFile, 100, "experiments/heaps/%s_heaps.csv", argv[1]);
	printf("%s\n", heapsFile);
	fo = fopen(heapsFile, "w");
	if(fo == NULL)
	{
		printf("No such file\n");
		exit(0);
	}

	for (i=0; i<seqLength; ++i)
	{
		fprintf(fo, "%ld,", heaps[i]);
	}

	fclose(fo);
	printf("Heaps computed\n");

	// ---------------------------------------------------------------------------------------
	// Taylor's Law	& Ebeling's Law
	// ---------------------------------------------------------------------------------------
	index=0;
	noOfSubSeq = (int)log10(seqLength);

	snprintf (taylorsFile, 100, "experiments/taylors/%s_taylors.csv", argv[1]);
	printf("%s\n", taylorsFile);
	fo = fopen(taylorsFile, "w");

	snprintf (ebelingsFile, 100, "experiments/ebelings/%s_ebelings.csv", argv[1]);
	printf("%s\n", ebelingsFile);
	fo1 = fopen(ebelingsFile, "w");

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

	for(i=10; i<seqLength-1; i=i*10)
	{
		printf("Computing Taylors Law for Subsequence Length: %lld\n", i);
		fprintf(fo, "%lld\n", i);
		tempSum = 0;
		subSeqLength[index] = i;

		mean = (double*) malloc(vocabularySize*sizeof(double));
		for(k=0; k<vocabularySize; k++)
		{
			mean[k]=0;
		}

		// Compute mean
		for(j=0; j<seqLength-i+1; j++)
		{
			for(k=j; k<j+i; k++)
			{
				mean[wordIDSeq[k]]++;
			}
		}

		for(k=0; k<vocabularySize; k++)
		{
			mean[k] = mean[k]/(double)(seqLength-i+1);
			fprintf(fo, "%f,", mean[k]);
		}
		fprintf(fo, "\n");
		printf("Computed Mean\n");

		// Compute variance
		sd = (double*) malloc(vocabularySize*sizeof(double));
		for(k=0; k<vocabularySize; k++)
		{
			sd[k]=0;
		}

		for(k=0; k<vocabularySize; k++)
		{
			wordFreqList[k]=0;
		}

		for(j=0; j<seqLength-i+1; j++)
		{
			if(j==0)
			{
				for(k=j; k<j+i; k++)
					wordFreqList[wordIDSeq[k]]++;
			}
			else
			{
				wordFreqList[wordIDSeq[j-1]]--;
				wordFreqList[wordIDSeq[j+i-1]]++;
			}

			for (k=0; k<vocabularySize; k++)
			{
				sd[k] += (wordFreqList[k]-mean[k])*(wordFreqList[k]-mean[k]);
			}
		}

		for (k=0; k<vocabularySize; k++)
		{
			sd[k] = sd[k]/(double)(seqLength-i+1);
			tempSum += sd[k];
			sd[k] = sqrt(sd[k]);
			fprintf(fo, "%f,", sd[k]);
		}
		fprintf(fo, "\n");
		fprintf(fo1, "%lld:%f,", i, tempSum);
		printf("Computed Standard Deviation\n");

		variance[index++] = tempSum;
	}
	fclose(fo);
	fclose(fo1);

	printf("Taylor's Law & Ebeling's Method computed\n");
}
