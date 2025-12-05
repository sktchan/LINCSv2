#!/bin/bash

#SBATCH --job-name=PE_rank_tf
#SBATCH --output=out/PE_rank_tf.out
#SBATCH --error=out/PE_rank_tf.err

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-gpu=32G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=2-00:00:00

export CUDA_VISIBLE_DEVICES=3

julia /home/golem/scratch/chans/lincsv2/improving_tf/newposenc_rank_tf.jl