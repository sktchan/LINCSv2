#!/bin/bash

#SBATCH --job-name=200_rank_tf
#SBATCH --output=out/200_rank_tf.out
#SBATCH --error=out/200_rank_tf.err

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-gpu=32G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=1-00:00:00

julia /home/golem/scratch/chans/lincsv2/improving_tf/200_rank_tf.jl