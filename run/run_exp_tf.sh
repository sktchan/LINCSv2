#!/bin/bash

#SBATCH --job-name=exp_tf
#SBATCH --output=out/exp_tf.out
#SBATCH --error=out/exp_tf.err

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G

#SBATCH --gres=gpu:V100:1
#SBATCH --time=2-00:00:00

julia /home/golem/scratch/chans/lincs/scripts/sept20/exp_tf.jl