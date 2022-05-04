#!/bin/bash
#SBATCH -n 16
#SBATCH -A IREL
#SBATCH -G 4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=10
#SBATCH --mail-user=samyak.ja@students.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=op_file_complex_qaTest.txt

echo "Activating virtualenv"
source activate base
#python3 /home2/samyak.ja/qg_incremental_training.py
cd M_EmbedKGQA/KGQA/RoBERTa
python main.py  --mode test --hops webqsp_full --nb_epochs 2 --model ComplEx --use_cuda True --outfile best_score_model_ComplEx_full --batch_size 4 --que_embedding_model SentenceTransformer
