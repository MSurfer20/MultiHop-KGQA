#!/bin/bash
#SBATCH -n 40
#SBATCH -A IREL
#SBATCH -G 4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=40
#SBATCH --mail-user=samyak.ja@students.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=op_file_relation_matching_hop2.txt
cd M_EmbedKGQA

#module load cudnn/7.6.5-cuda-10.2
#module load cudnn/8.2.1-cuda-11.3
#module load python/3.8.3

#python3 -m venv elmo_venv
#source ./elmo_venv/bin/activate

#python3 -m pip install --upgrade --no-cache-dir wheel
#python3 -m pip install -r requirements.txt

#cd train_embeddings
#git clone https://github.com/uma-pi1/kge.git && cd kge
#cd kge
#pip install -e .
#cd ../
#pwd
#kge start ./config/complex-train-webqsp-full.yaml --search.device_pool cuda:0,cuda:1,cuda:2,cuda:3
cd KGQA/Roberta_EmbedKGQA
#python main.py  --mode train --load_from best_score_model --hops webqsp_full --relation_dim 128 --patience 20 --nb_epochs 200 --model ComplEx --use_cuda True --outfile best_score_model_complex_roberta_full --batch_size 16 --que_embedding_model RoBERTa

python relation_matching.py
