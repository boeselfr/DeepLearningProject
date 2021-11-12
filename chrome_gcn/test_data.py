import os
import csv


path = '/Users/fredericboesel/Documents/master/herbst21/deeplearning/data/processed_data/GM12878/1000/'

bin_dict = {}
chroms = ['chr1', 'chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12']
for chrom in chroms:
	bin_dict[chrom]={}

with open(os.path.join(path, 'windows.bed')) as csvfile:
    csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','assay_id','score','strand','signalValue','pvalue','qValue','peak'])
    for csv_row in csv_reader:
        chrom = str(csv_row['chrom'])
        start_pos = int(csv_row['start_pos'])
        assay_id = str(csv_row['assay_id'])

        if chrom in chroms:
            if start_pos not in bin_dict[chrom]:
                bin_dict[chrom][start_pos] = {}

for chrom in chroms:
    print(f'chrom: {chrom}')
    print(f'len of bin dict: {len(bin_dict[chrom])}')





