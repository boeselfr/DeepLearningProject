# Deep Learning Project

Predicting splicing behaviour based on combined local sequence and long-range 3D genome information.

## Installation

Navigate to src folder, then run:

pip install -e splicing/

## Data Pipeline

First, check config.yaml and set data directory as well as data pipeline params.

Then, run:
python preprocessing/create_datafile.py -g train -p all
python preprocessing/create_datafile.py -g test -p 0

python preprocessing/create_dataset.py -g train -p all
python preprocessing/create_dataset.py -g test -p 0