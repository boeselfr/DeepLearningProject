import os
import argparse
import yaml

create_windows = __import__('gcn_1create_windows')
#create_peaks = __import__('2create_peaks')
#create_windows_with_peaks = __import__('3create_windows_with_peaks')
#create_seqs = __import__('4create_seqs')
#merge_seqs_and_labels = __import__('5merge_seqs_and_labels')
#create_input_label_files = __import__('6create_input_label_files')
create_graph = __import__('gcn_7create_graph_new')
# create_graph = __import__('7create_graph_old')

###############################################################################
# Loading Config
###############################################################################

with open("config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

DATA_DIR = config['DATA_DIRECTORY']

GENOME_ROOT = os.path.join(
    config['DATA_DIRECTORY'], 
    config['RAW_DATA']['genome_root']
)

INPUT_ROOT = os.path.join(
    config['DATA_DIRECTORY'], 
    config['RAW_DATA']['input_root']
)

HIC_ROOT = os.path.join(
    config['DATA_DIRECTORY'], 
    config['RAW_DATA']['hic_root']
)

EXPR_ROOT = os.path.join(
    config['DATA_DIRECTORY'], 
    config['RAW_DATA']['expr_root']
)

OUTPUT_ROOT = os.path.join(
    config['DATA_DIRECTORY'], 
    config['DATA_PIPELINE']['output_dir']
)

INTERVAL = config['DATA_PIPELINE']['window_size']

CONTEXT_LENGTH = config['DATA_PIPELINE']['context_length']

###############################################################################
# Parsing Args
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--run_file', type=str, choices=['1','2','3','4','5','6','7','all'], default='1')
parser.add_argument('--genome', type=str, default='hg19')
parser.add_argument('--cell_type', type=str, default='GM12878')
parser.add_argument('--window_length', type=int, default=INTERVAL)
parser.add_argument('--extended_window_length', type=int, default=INTERVAL) # can change this
parser.add_argument('--genome_root', type=str, default=GENOME_ROOT)
parser.add_argument('--input_root', type=str,  default=INPUT_ROOT)
parser.add_argument('--hic_root', type=str,    default=HIC_ROOT)
parser.add_argument('--expr_root', type=str,   default=EXPR_ROOT)
parser.add_argument('--output_root', type=str, default=OUTPUT_ROOT)
parser.add_argument('--use_all_windows', action='store_true',help='use all windows, otherwise use only the windows that contain genes')
parser.add_argument('--norm', type=str, choices=['','KR','VC','SQRTVC'], default='SQRTVC')
parser.add_argument('--resolution', type=str, default='5')
parser.add_argument('--hic_edges', type=int, default=500000)
parser.add_argument('--min_distance_threshold', type=int, default=1000)
parser.add_argument('--context_length', type=int, default=CONTEXT_LENGTH, help='context length the window.bed file got created with')
args = parser.parse_args()

# make sure that window size and resolution match up:
try:
    args.resolution = str(int(args.window_length / 1000))
except Exception as e:
    print(f'the specified window size {args.window_length} is not valid!')

args.stride_length=args.window_length
args.chrom_sizes = os.path.join(args.genome_root,args.genome,args.genome+'.chrom_sizes')
args.genome_fasta = os.path.join(args.genome_root,args.genome,args.genome+'.fa')
args.tad_file = os.path.join(args.genome_root,args.genome,args.genome+'.TADs',args.cell_type+'_Lieberman-raw_TADs.txt')
#args.output_root = os.path.join(args.output_root ,args.cell_type,str(args.window_length))

if not os.path.exists(args.output_root):
    os.makedirs(args.output_root)

#TODO: align chromosome handling b/w the two, which ones are train/test
# e.g. args.train_chroms = config['DATA_PIPELINE']['train_chroms']
args.chroms=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10',
             'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19',
             'chr20','chr21','chr22']

args.valid_chroms = ['chr3', 'chr12', 'chr17']
args.test_chroms = ['chr1', 'chr8', 'chr21']

if args.cell_type == 'GM12878':
    args.residuals = [0]
else:
    args.residuals = [0,1000,2000,3000,4000]

    # args.min_distance_threshold=5000

if args.run_file in ['1','all']:
    print('1create_windows')
    create_windows.create_windows(args)
if args.run_file in ['7']:
    print('7create_graph')
    create_graph.create_graph(args)


