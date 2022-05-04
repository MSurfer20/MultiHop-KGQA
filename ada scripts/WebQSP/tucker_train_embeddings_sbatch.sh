#!/bin/bash
#SBATCH -n 16
#SBATCH -A IREL
#SBATCH -G 4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=10
#SBATCH --mail-user=samyak.ja@students.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=op_file.txt

echo "Activating virtualenv"
source activate base
#python3 /home2/samyak.ja/qg_incremental_training.py
cd M_EmbedKGQA
kge resume ./kge/local/experiments/20220317-063017-tucker-train-webqsp-full/config.yaml --search.device_pool cuda:0,cuda:1,cuda:2,cuda:3
