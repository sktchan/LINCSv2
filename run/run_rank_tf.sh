#!/bin/bash

#SBATCH --job-name=rank_tf
#SBATCH --output=out/rank_tf.out
#SBATCH --error=out/rank_tf.err

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=2-00:00:00

julia /home/golem/scratch/chans/lincs/scripts/sept20/rank_tf.jl