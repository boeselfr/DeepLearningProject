CL_max=80  # original 10000
# Maximum nucleotide context length (CL_max/2 on either side of the 
# position of interest)
# CL_max should be an even number

SL=256  # original 5000
# Sequence length of SpliceAIs (SL+CL will be the input length and
# SL will be the output length)

# Input details
data_dir='/home/anej/repos/studies/deep-learning-project/data/'

splice_table='/home/anej/repos/studies/deep-learning-project/data/gtex_dataset.txt'
ref_genome='/home/anej/repos/studies/deep-learning-project/data/genome.fa'

# Output details
sequence='/home/anej/repos/studies/deep-learning-project/data/gtex_sequence.txt'
