#!/bin/bash
#
#SBATCH --job-name=Level
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=40


module purge

cd ../../
source venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.9/bin:$PATH
source ~/.bashrc

cd Post-Prediction-Imputation
python3 Level.py 