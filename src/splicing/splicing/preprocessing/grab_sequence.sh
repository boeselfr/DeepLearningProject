#!/bin/bash

#!/bin/sh

# include parse_yaml function
. preprocessing/parse_yaml.sh

# read yaml file
eval $(parse_yaml config.yaml "config_")

# access yaml content

CL_max=$config_SPLICEAI_cl_max
SL=$config_SPLICEAI_sl

#declare -i CL_max=80  # original 10000
# Maximum nucleotide context length (CL_max/2 on either side of the 
# position of interest)
# CL_max should be an even number

#declare -i SL=5000  # original 5000
# Sequence length of SpliceAIs (SL+CL will be the input length and
# SL will be the output length)

# Input details
data_dir=$config_DATA_DIRECTORY/$config_SPLICEAI_data

echo data_dir

splice_table=$data_dir/$config_SPLICEAI_splice_table

echo splice_table

ref_genome=$data_dir/$config_SPLICEAI_genome

# Output details
sequence=$data_dir/$config_SPLICEAI_sequence
echo creating sequence file $sequence

CLr=$((CL_max/2))
CLl=$(($CLr+1))
# First nucleotide not included by BEDtools

cat $splice_table | awk -v CLl=$CLl -v CLr=$CLr '{print $3"\t"($5-CLl)"\t"($6+CLr)}' > temp.bed

bedtools getfasta -bed temp.bed -fi $ref_genome -fo $sequence -tab

rm temp.bed
