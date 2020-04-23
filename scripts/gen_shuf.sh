#!/bin/bash
cd ../Data
chromosomes=({1..22} M X Y)

# Load chromosome sizes
#curl -O "https://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.chrom.sizes"

# Generate shuffled fix chromosome, exclude quadruplexes
bedtools shuffle -i G4_chip.bed -g hg19.chrom.sizes -chrom -excl G4_chip.bed -seed 42 > shuffled_full.bed

# Filter quadruplexes by positions
for num in ${chromosomes[*]}
do
	if [ -f chr$num.fa ]
	then	
		bedtools getfasta -fi chr$num.fa -bed shuffled_full.bed -bedOut -tab > shuffled_chr$num.bed
	else
		echo "Couldn't find chr$num.fa"
	fi
done


