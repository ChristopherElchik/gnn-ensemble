#!/bin/bash
#BSUB -n 1
#BSUB -W 120
#BSUB -q gpu
#BSUB -R "select[h100]"
#BSUB -gpu "num=1:mode=shared:mps=yes:gmem=2GB"
#BSUB -o out.%J
#BSUB -e err.%J

cd /share/csc591038f25/cwelchik/hw1/tunedGNN
source venv/bin/activate
cd large_graph/data/ogb/ogbn_arxiv
rm -rf processed
mkdir processed
cd /share/csc591038f25/cwelchik/hw1/tunedGNN/large_graph

# Check GPU availability
nvidia-smi

# Run your GNN training
export TORCH_LOAD_WEIGHTS_ONLY=False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 512 --epochs 2000 --lr 0.0005 --runs 1 --local_layers 3 --bn --device 0 --res --batch_size 100000

