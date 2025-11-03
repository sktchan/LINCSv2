#!/bin/bash

#SBATCH --account=def-lemieux1
#SBATCH --job-name=rank_tf
#SBATCH --output=out/rank_tf.out
#SBATCH --error=err/rank_tf.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=62G
#SBATCH --gres=gpu:H100-3g.40gb:1

#SBATCH --time=4-:00:00

#SBATCH --mail-user=serenaktc@gmail.com
#SBATCH --mail-type=ALL

# If SLURM_SUBMIT_DIR is not set (manual run), fallback to current directory
if [ -z $SLURM_SUBMIT_DIR ]; then
    export SLURM_SUBMIT_DIR=$(pwd)
fi

cd $SLURM_SUBMIT_DIR

module load julia

julia /home/chans/links/scratch/lincs/rank_tf.jl