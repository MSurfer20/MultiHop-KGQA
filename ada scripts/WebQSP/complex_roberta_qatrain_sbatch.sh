#!/bin/bash
#SBATCH -n 16
#SBATCH -A IREL
#SBATCH -G 4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=10
#SBATCH --mail-user=samyak.ja@students.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=op_file_tuckER_qaTrain.txt

echo "Activating virtualenv"
source activate base
#python3 /home2/samyak.ja/qg_incremental_training.py
cd M_EmbedKGQA/KGQA/RoBERTa
python main.py  --mode train --hops webqsp_full --load_from best_score_model --relation_dim 200 --patience 20 --nb_epochs 200 --model ComplEx --use_cuda True --outfile best_score_model_complex_roberta_full --batch_size 16 --que_embedding_model RoBERTa

#python main.py  --mode train --hops webqsp_full --relation_dim 200 --nb_epochs 5 --model TuckER --use_cuda True --outfile best_score_model_tuckER_full --batch_size 16 --que_embedding_model RoBERTa
