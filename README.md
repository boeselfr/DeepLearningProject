# Deep Learning Project

Predicting splicing behaviour based on combined local sequence and long-range 3D genome information.

## Repo Setup and Package Installation

First, please clone the repo in Euler. Then, navigate to main folder and run 'source ./init_leonhard.sh'

Then, install torch-geometry as explained here: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Then, navigate to src folder and run:

pip install -e splicing/

## Important: Structure of Repository and Commands
The entire project revolves around two required arguments for every command:
- Window size (-ws): 1000 or 5000
- Context length (-cl): 80 or 400

At every step, from data preprocessing to model training, these arguments are required.
Separate datasets, models, etc. are created for each combination of these settings.

The rest of this guide takes you through the -ws 5000 -cl 80 workflow, just change the values
to run for a different data configuration.

## Data Pipeline

### Setup
First, create a data directory for the project. Then, download raw data from <insert link>,
and extract the "raw_data" folder within the data directory.

Then, open src/splicing/splicing/config.yaml and set the DATA_DIRECTORY variable at the top
to match the path of the directory you just created.

### Euler Commands

all commands will be run from src/splicing/splicing

bsub -R "rusage[mem=16000]" python preprocessing/create_datafile.py -g train -p all -ws 5000 -cl 80

bsub -R "rusage[mem=16000]" python preprocessing/create_datafile.py -g valid -p all -ws 5000 -cl 80

bsub -R "rusage[mem=16000]" python preprocessing/create_datafile.py -g test -p all -ws 5000 -cl 80

bsub -R "rusage[mem=16000]" python preprocessing/create_dataset_nodup.py -g train -p all -ws 5000 -cl 80

bsub -R "rusage[mem=16000]" python preprocessing/create_dataset_nodup.py -g valid -p all -ws 5000 -cl 80

bsub -R "rusage[mem=16000]" python preprocessing/create_dataset_nodup.py -g test -p all -ws 5000 -cl 80

### Graph Creation
You now need to create the graph using the hic values that need to be downloaded before.

The hic data can be downloaded here:
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525

filename: GSE63525_GM12878_combined_intrachromosomal_contact_matrices.tar.gz
(Gene Expression Omnibus (GEO) accession number for the data sets reported in this paper is GSE63525.)

The hic datafolder 5kb_resolution_intrachromosomal (for 5000 window size) or 1kb_resolution_intrachromosomal (for 1000 window size)
needs to be located here: 
data_dir/raw_data/hic/GM12878_combined/

After the download is complete, run the following command to create the graph:

bsub -R "rusage[mem=32000]" -W 4:00 python preprocessing/create_graph.py -ws 5000 -cl 80

The command relies on the file graph_windows_5000_80.bed, that is created in the create_datafile command from above.




## Model Training

### Pretrain the nucleotide representation CNN
bsub -R "rusage[mem=48000,ngpus_excl_p=1]" -W 04:00 python main.py -pretrain -modelid base -ws 5000 -cl 80 -wb

-modelid enables you to train multiple models with the same -ws and -cl without overwriting.
-wb enables the WandB logging.

### Export nucleotide representations using saved pretrained model
bsub -R "rusage[mem=64000,ngpus_excl_p=1]" python main.py -save_feats -load_pretrained -modelid base -mit 10 -ws 5000 -cl 80

-mit specifies which model iteration (epoch) to load. Default number of epochs for pretraining is 10.

### Train Graph & Full model
bsub -R "rusage[mem=64000,ngpus_excl_p=1]" -W 24:00 python main.py -finetune -modelid base -ws 5000 -cl 80 -adj_type both -test -wb -wbn default_both
bsub -R "rusage[mem=64000,ngpus_excl_p=1]" -W 24:00 python main.py -finetune -modelid base -ws 5000 -cl 80 -adj_type hic -test -wb -wbn default_hic
bsub -R "rusage[mem=64000,ngpus_excl_p=1]" -W 24:00 python main.py -finetune -modelid base -ws 5000 -cl 80 -adj_type constant -test -wb -wbn default_constant
bsub -R "rusage[mem=64000,ngpus_excl_p=1]" -W 24:00 python main.py -finetune -modelid base -ws 5000 -cl 80 -adj_type none -test -wb -wbn default_none

- adj_type: specifies which type of graph to use: hic, constant, both, none
- test: predicts on test set at the end of each epoch
- wbn: name of the run in WandB