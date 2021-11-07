import os

CL_max=400  # original 10000
# Maximum nucleotide context length (CL_max/2 on either side of the 
# position of interest)
# CL_max should be an even number

SL=5000  # original 5000
# Sequence length of SpliceAIs (SL+CL will be the input length and
# SL will be the output length)

# Input details
data_dir=os.path.join(os.environ['SOURCE'], "dl_data/spliceai")

splice_table=os.path.join(data_dir, 'gtex_dataset.txt')
ref_genome=os.path.join(data_dir, 'genome.fa')

# Output details
sequence=os.path.join(data_dir, 'gtex_sequence.txt')

# module load eth_proxy gcc/8.2.0 python_gpu cudnn/8.1.0.77 bedtools2
