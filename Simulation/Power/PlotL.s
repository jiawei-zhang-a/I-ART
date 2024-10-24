#!/bin/bash
#
#SBATCH --job-name=PlotL
#SBATCH --nodes=1
#SBATCH --time=19:19:00
#SBATCH --mem=250MB
#SBATCH --cpus-per-task=1
#SBATCH --output=Plot/L.out
#SBATCH --error=Plot/L.err

export OMP_NUM_THREADS=1

module purge

singularity exec --overlay /scratch/jz4721/pyenv/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/jz4721/Post-prediction-Causal-Inference/Simulation/Visulization/; python3 /scratch/jz4721/Post-prediction-Causal-Inference/Simulation/Visulization/PlotLConvergence.py"
