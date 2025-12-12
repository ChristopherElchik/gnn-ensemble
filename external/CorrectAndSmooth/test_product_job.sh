#!/bin/bash
#BSUB -n 1
#BSUB -W 120
#BSUB -q gpu
#BSUB -R "select[l40]"
#BSUB -gpu "num=1"
#BSUB -o out.%J
#BSUB -e err.%J

cd /share/csc591038f25/cwelchik/proj/CorrectAndSmooth
source venv/bin/activate
# cd large_graph/data/ogb/ogbn_products
# rm -rf processed
# mkdir processed
# cd /share/csc591038f25/cwelchik/hw1/tunedGNN/large_graph

# Check GPU availability
nvidia-smi

# Run your GNN training
export TORCH_LOAD_WEIGHTS_ONLY=False
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load julia
export JULIA_DEPOT_PATH="$PWD/.julia_depot"

julia -e 'using Pkg; Pkg.add(["PyCall", "LinearMaps", "Arpack"])'
# python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 512 --epochs 10 --lr 0.0005 --runs 1 --local_layers 5 --bn --device 0 --res --batch_size 100000
# python product.py --device 0 --ln --hidden_channels 64 --gnn gcn --epochs 300 --save_model gcn_model.pt --save_predictions gcn_predictions.csv
# python gen_models.py --dataset products --model linear --use_embeddings --epochs 500 --lr 0.05 --runs 2
# python gen_models.py --dataset products --model linear --use_embeddings --epochs 1000 --lr 0.1
python run_experiments.py --dataset products --method linear
