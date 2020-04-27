#!/bin/bash
cd ../Data
chromosomes=({1..22} M X Y)


for num in ${chromosomes[*]}
do
	# Download fasta file for chromosome
	echo 'chr'$num
	curl -O "ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/chr$num.fa.gz"
	
	if [ -f chr$num.fa.gz ]
	then
		gunzip chr$num.fa.gz	
		# Filter quadruplexes by positions
		bedtools getfasta -fi chr$num.fa -bed G4_chip_centered.bed -bedOut -tab > chr${num}_centered.bed 2> warnings.txt
	else
		echo "Couldn't find chr$num.fa"
	fi
done

#cat *.fa > full.fa

