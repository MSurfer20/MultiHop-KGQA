#!/bin/bash
#SBATCH -n 40
#SBATCH -A IREL
#SBATCH -G 4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=40
#SBATCH --mail-user=samyak.ja@students.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=op_file_tuckER_sbert_qaTrain.txt

echo "Activating virtualenv"
source activate base
#python3 /home2/samyak.ja/qg_incremental_training.py
cd M_EmbedKGQA/KGQA/RoBERTa
#python main.py  --mode train --hops webqsp_full --load_from best_score_model --relation_dim 200 --patience 20 --nb_epochs 200 --model TuckER --use_cuda True --lr 1e-5 --outfile best_score_model_tuckER_full --batch_size 4 --que_embedding_model RoBERTa

#python main.py  --mode train --hops webqsp_full --relation_dim 200 --nb_epochs 5 --model TuckER --use_cuda True --outfile best_score_model_tuckER_full --batch_size 16 --que_embedding_model RoBERTa

python3 main.py --mode train --hops webqsp_full --relation_dim 200 --do_batch_norm 1 --freeze 1 --batch_size 16 --validate_every 10 --lr 0.00002 --entdrop 0.0 --reldrop 0.0 --scoredrop 0.0 --decay 1.0 --model TuckER --patience 20 --ls 0.05 --l3_reg 0.001 --nb_epochs 200 --outfile best_score_model_tuckER_full --que_embedding_model SentenceTransformer
