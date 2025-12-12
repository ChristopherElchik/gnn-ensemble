#!/bin/bash
#BSUB -n 1
#BSUB -W 240
#BSUB -q gpu
#BSUB -R "select[a100]"
#BSUB -gpu "num=1"
#BSUB -o out.%J
#BSUB -e err.%J

cd /share/csc591038f25/cwelchik/hw1/tunedGNN
source venv/bin/activate
cd large_graph/data/ogb/ogbn_products
rm -rf processed
mkdir processed
cd /share/csc591038f25/cwelchik/hw1/tunedGNN/large_graph

# Check GPU availability
nvidia-smi

# Run your GNN training
export TORCH_LOAD_WEIGHTS_ONLY=False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 512 --epochs 10 --lr 0.0005 --runs 1 --local_layers 5 --bn --device 0 --res --batch_size 100000
python product.py --device 0 --ln --gnn gcn --save_outputs gcn_results_a100.pt
