#!/bin/bash
#SBATCH --partition=main  #unkillable #main #long
#SBATCH --output=tgn_lastfmgenre_s1.txt 
#SBATCH --error=tgn_lastfmgenre_s1_error.txt   
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:rtx8000:1                  # Ask for 1 titan xp
#SBATCH --mem=32G                             # Ask for 32 GB of RAM
#SBATCH --time=48:00:00                       # The job will run for 1 day

export HOME="/home/mila/h/huangshe"
module load python/3.9
source $HOME/tgbenv/bin/activate

pwd
CUDA_VISIBLE_DEVICES=0 python examples/nodeproppred/lastfmgenre/tgn.py -s 1
