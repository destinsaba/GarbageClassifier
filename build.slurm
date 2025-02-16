#! /bin/bash
# ============================
#SBATCH --job-name=ENEL645-A2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=job_output.log
# ==================================

source ~/software/init-conda
conda activate pytorch

python /home/destin.saba/transfer-learning/garbageClassifer.py