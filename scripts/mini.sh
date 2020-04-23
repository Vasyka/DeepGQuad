#!/bin/bash
cd ../Data

a=({1..22} M X Y)

for var in ${a[*]} 
do
	echo $var
done


find . -name "chr[${a[*]}].fa"

