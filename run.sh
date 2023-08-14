#!/bin/bash
#SBATCH --partition=long  #unkillable #main #long
#SBATCH --output=tgnlog_genre_s5.txt #tgn_lastfmgenre_s5.txt 
#SBATCH --error=tgnlog_genre_s5error.txt #tgn_lastfmgenre_s5_error.txt   
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 titan xp
#SBATCH --mem=32G                             # Ask for 32 GB of RAM
#SBATCH --time=48:00:00                       # The job will run for 1 day

export HOME="/home/mila/h/huangshe"
module load python/3.9
source $HOME/tgbenv/bin/activate

pwd
CUDA_VISIBLE_DEVICES=0 python examples/nodeproppred/tgbn-genre/tgn.py --seed 5
# CUDA_VISIBLE_DEVICES=0 python examples/nodeproppred/lastfmgenre/dyrep.py --seed 5
# CUDA_VISIBLE_DEVICES=0 python examples/nodeproppred/un_trade/tgn.py -s 5
# CUDA_VISIBLE_DEVICES=0 python examples/linkproppred/amazonreview/tgn.py -s 1
