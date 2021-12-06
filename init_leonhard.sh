#!/bin/bash

module load eth_proxy gcc/8.2.0 python_gpu cudnn/8.1.0.77 bedtools2

if [ ! -d "venv/" ]; then
    python3 -m venv venv
    echo "Created virtual environment."
fi

export TA_CACHE_DIR="/scratch/$USER/.cache"

source venv/bin/activate
pip3 install --default-timeout=100 -r requirements.txt