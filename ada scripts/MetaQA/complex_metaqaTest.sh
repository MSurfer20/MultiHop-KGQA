#!/bin/bash
#SBATCH -n 40
#SBATCH -A IREL
#SBATCH -G 4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=40
#SBATCH --mail-user=samyak.ja@students.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=op_file_complex_metaqaTest_3hop.txt

echo "Activating virtualenv"
source activate base
#python3 /home2/samyak.ja/qg_incremental_training.py
cd M_EmbedKGQA/KGQA/LSTM
python main.py --mode test --hops 3 --model ComplEx --use_cuda True --kg_type full --hidden_dim 256 --relation_dim 200
