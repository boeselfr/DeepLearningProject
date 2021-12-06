#!/bin/bash

module load eth_proxy gcc/8.2.0 python_gpu cudnn/8.1.0.77 bedtools2

#if [ ! -d "venv/" ]; then
#    python3 -m venv venv
#    echo "Created virtual environment."
#fi
#
#export TA_CACHE_DIR="/scratch/$USER/.cache"

#source venv/bin/activate
#pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
#pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
#pip3 install torch-geometric
#pip3 install --default-timeout=100 -r requirements.txt