import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from dataloader import DatasetMetaQA, DataLoaderMetaQA
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd
from text_process import create_embedding_dics, parse_text_file, get_vocab, data_generator

parser = argparse.ArgumentParser()
parser.add_argument('--hops', type=str, default='1') #Parameter to find the number of hops to be used
parser.add_argument('--ls', type=float, default=0.0) #Parameter for label smoothing(Pytorch)
parser.add_argument('--validate_every', type=int, default=5) #Parameter to do validation after every validate_every epoch
parser.add_argument('--model', type=str, default='TuckER') #Parameter to embedding model that is to be used(Tucker/Simple) 
parser.add_argument('--kg_type', type=str, default='half') #Parameter for whether half or full dataset is to be used

parser.add_argument('--mode', type=str, default='eval') #Mode for whether the code is to be run in train/test/eval mode
parser.add_argument('--batch_size', type=int, default=1024) #The batch size that is used during the model training
parser.add_argument('--dropout', type=float, default=0.1) #Dropout rate for training model
parser.add_argument('--entdrop', type=float, default=0.0) #Dropout rate for entitites
parser.add_argument('--reldrop', type=float, default=0.0) #Dropout rate for relations
parser.add_argument('--scoredrop', type=float, default=0.0) #Dropout rate for score
parser.add_argument('--l3_reg', type=float, default=0.0) #L3 regularization
parser.add_argument('--decay', type=float, default=1.0)#Learning rate decay
parser.add_argument('--num_workers', type=int, default=15)#Number of workers being used for reading the dataset
parser.add_argument('--lr', type=float, default=0.0001)#Learning rate for the model
parser.add_argument('--nb_epochs', type=int, default=90)#Total number of epochs
parser.add_argument('--hidden_dim', type=int, default=200)#Dimension of hidden layer
parser.add_argument('--embedding_dim', type=int, default=256)#Dimension of embedding layer
parser.add_argument('--relation_dim', type=int, default=30)#Dimension for relation embeddings
parser.add_argument('--patience', type=int, default=5)#The number of epochs that model waits when no improvement in accuracy
parser.add_argument('--freeze', type=bool, default=True)# Parameter that controls freezing of batch norm layers

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
args = parser.parse_args()


def inTopk(scores, ans, k):
    result = False
    topk = torch.topk(scores, k)[1]
    for x in topk:
        if x in ans:
            result = True
    return result

def validate(data_path, device, model, word2idx, entity2idx, model_name, return_hits_at_k):
    model.eval()
    data = parse_text_file(data_path)
    answers = []
    data_gen = data_generator(data=data, word2ix=word2idx, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0

    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0

    for i in tqdm(range(len(data))):
        try:
            d = next(data_gen)
            head = d[0].to(device)
            question = d[1].to(device)
            ans = d[2]
            ques_len = d[3].unsqueeze(0)
            tail_test = torch.tensor(ans, dtype=torch.long).to(device)

            scores = model.get_score_ranked(head=head, sentence=question, sent_len=ques_len)[0]
            mask = torch.zeros(len(entity2idx)).to(device)
            mask[head] = 1
            #reduce scores of all non-candidates
            new_scores = scores - (mask*99999)
            pred_ans = torch.argmax(new_scores).item()
            if pred_ans == head.item():
                print('Head and answer same')
                print(torch.max(new_scores))
                print(torch.min(new_scores))
            
            if inTopk(new_scores, ans, 1):
                hit_at_1 += 1
            if inTopk(new_scores, ans, 5):
                hit_at_5 += 1
            if inTopk(new_scores, ans, 10):
                hit_at_10 += 1


            if type(ans) is int:
                ans = [ans]
            is_correct = 0
            if pred_ans in ans:
                total_correct += 1
                is_correct = 1
            else:
                num_incorrect += 1
            q_text = d[-1]
            answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))
        except:
            error_count += 1
        
    accuracy = total_correct/len(data)

    if return_hits_at_k:
        return answers, accuracy, (hit_at_1/len(data)), (hit_at_5/len(data)), (hit_at_10/len(data))
    else:
        return answers, accuracy


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()


def get_checkpoint_file_path(chkpt_path, model_name, num_hops, suffix, kg_type):
    return f"{chkpt_path}{model_name}_{num_hops}_{suffix}_{kg_type}"
        
def perform_experiment(data_path, mode, entity_path, relation_path, entity_dict, relation_dict, batch_size, num_workers, nb_epochs, embedding_dim, hidden_dim, relation_dim,patience, freeze, validate_every, num_hops, lr, entdrop, reldrop, scoredrop, l3_reg, model_name, decay, ls, w_matrix, bn_list, kg_type, valid_data_path=None, test_data_path=None):
    entities = np.load(entity_path)
    relations = np.load(relation_path)
    
    #dict mapping entity/rel to entity/rel id -> e,r
    e = dict()
    r = dict()

    with open(entity_dict, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            ent_id = int(line[0])
            ent_name = line[1]
            e[ent_name] = entities[ent_id]

    with open(relation_dict,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            rel_id = int(line[0])
            rel_name = line[1]
            r[rel_name] = relations[rel_id]

    entity2idx, idx2entity, embedding_matrix = create_embedding_dics(e) #Entities->ID id->entity, embed matrix->array of all IDs
    data = parse_text_file(data_path, split=False)
    word2ix,idx2word, max_len = get_vocab(data)
    hops = str(num_hops)
    device = torch.device('cuda')

    dataset = DatasetMetaQA(data=data, word2ix=word2ix, relations=r, entities=e, entity2idx=entity2idx)

    model = RelationExtractor(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(word2ix), num_entities = len(idx2entity), relation_dim=relation_dim, pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop = entdrop, reldrop = reldrop, scoredrop = scoredrop, l3_reg = l3_reg, model = model_name, ls = ls, w_matrix = w_matrix, bn_list=bn_list)

    checkpoint_path = '../../checkpoints/MetaQA/'
    if mode=='train':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, decay)
        optimizer.zero_grad()
        model.to(device)
        best_score = -float("inf")
        best_model = model.state_dict()
        no_update = 0
        data_loader = DataLoaderMetaQA(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for epoch in range(nb_epochs):
            phases = []
            for i in range(validate_every):
                phases.append('train')
            phases.append('valid')
            for phase in phases:
                if phase == 'train':
                    model.train()
                    if freeze == True:
                        # print('Freezing batch norm layers')
                        model.apply(set_bn_eval)
                    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                    running_loss = 0
                    for i_batch, a in enumerate(loader):
                        model.zero_grad()
                        question = a[0].to(device)
                        sent_len = a[1].to(device)
                        positive_head = a[2].to(device)
                        positive_tail = a[3].to(device)                    

                        loss = model(sentence=question, p_head=positive_head, p_tail=positive_tail, question_len=sent_len)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        loader.set_postfix(Loss=running_loss/((i_batch+1)*batch_size), Epoch=epoch)
                        loader.set_description('{}/{}'.format(epoch, nb_epochs))
                        loader.update()
                    
                    scheduler.step()

                elif phase=='valid':
                    model.eval()
                    eps = 0.0001
                    answers, score = validate(model=model, data_path= valid_data_path, word2idx= word2ix, entity2idx= entity2idx, device=device, model_name=model_name, return_hits_at_k=False)
                    if score > best_score + eps:
                        best_score = score
                        no_update = 0
                        best_model = model.state_dict()
                        print(hops + " hop Validation accuracy increased from previous epoch", score)
                        _, test_score = validate(model=model, data_path= test_data_path, word2idx= word2ix, entity2idx= entity2idx, device=device, model_name=model_name, return_hits_at_k=False)
                        print('Test score for best valid so far:', test_score)
                        suffix = ''
                        if freeze == True:
                            suffix = '_frozen'
                        checkpoint_file_name = get_checkpoint_file_path(checkpoint_path, model_name, num_hops, suffix, kg_type)+'.chkpt'
                        print('Saving checkpoint to ', checkpoint_file_name)
                        torch.save(model.state_dict(), checkpoint_file_name)
                    elif (score < best_score + eps) and (no_update < patience):
                        no_update +=1
                        print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, patience-no_update))
                    elif no_update == patience:
                        print("Model has exceed patience. Saving best model and exiting")
                        torch.save(best_model, get_checkpoint_file_path(checkpoint_path, model_name, num_hops, '', kg_type)+ '_' + 'best_score_model.chkpt')
                        exit()
                    if epoch == nb_epochs-1:
                        print("Final Epoch has reached. Stopping and saving model.")
                        torch.save(best_model, get_checkpoint_file_path(checkpoint_path, model_name, num_hops, '', kg_type)+ '_' + 'best_score_model.chkpt')
                        exit()
    elif mode=='test':
        model_chkpt_file=get_checkpoint_file_path(checkpoint_path, model_name, num_hops, '', kg_type)+ '_' + 'best_score_model.chkpt'
        
        print(model_chkpt_file)
        
        model.load_state_dict(torch.load(model_chkpt_file, map_location=lambda storage, loc: storage))
        model.to(device)

        answers, accuracy, hits_at_1, hits_at_5, hits_at_10  = validate(model=model, data_path= test_data_path, word2idx= word2ix, entity2idx= entity2idx, device=device, model_name=model_name, return_hits_at_k=True)

        d = {
            'KG-Model': model_name,
            'KG-Type': kg_type,
            'hops': num_hops,
            'Accuracy': [accuracy], 
            'Hits@1': [hits_at_1],
            'Hits@5': [hits_at_5],
            'Hits@10': [hits_at_10]
            }
        df = pd.DataFrame(data=d)
        df.to_csv(f"final_results.csv", mode='a', index=False, header=False)       
                    

hops = args.hops
if hops not in ['1', '2', '3']:
    print("Reduce number of hops")
    exit(0)

valid_data_path = '../../data/QA_data/MetaQA/qa_dev_' + hops + 'hop' + '.txt'
test_data_path = '../../data/QA_data/MetaQA/qa_test_' + hops + 'hop' + '.txt'
if args.kg_type == 'half':
    data_path = '../../data/QA_data/MetaQA/qa_train_' + hops + 'hop' + '_half.txt'
else:
    data_path = '../../data/QA_data/MetaQA/qa_train_' + hops + 'hop' + '.txt'


model_name = args.model
kg_type = args.kg_type
embedding_folder = '../../pretrained_models/embeddings/' + model_name + '_MetaQA_' + kg_type

entity_embedding_path = embedding_folder + '/E.npy'
relation_embedding_path = embedding_folder + '/R.npy'
entity_dict = embedding_folder + '/entities.dict'
relation_dict = embedding_folder + '/relations.dict'
w_matrix =  embedding_folder + '/W.npy'
print('KG type: ', kg_type)

bn_list = []

for i in range(3):
    bn = np.load(embedding_folder + '/bn' + str(i) + '.npy', allow_pickle=True)
    bn_list.append(bn.item())

perform_experiment(data_path=data_path, 
mode=args.mode,
bn_list=bn_list,
entity_path=entity_embedding_path, 
test_data_path=test_data_path,
entity_dict=entity_dict, 
batch_size=args.batch_size,
nb_epochs=args.nb_epochs, 
freeze=args.freeze,
embedding_dim=args.embedding_dim, 
validate_every=args.validate_every,
hidden_dim=args.hidden_dim, 
relation_dim=args.relation_dim, 
valid_data_path=valid_data_path,
num_workers=args.num_workers,
l3_reg = args.l3_reg,
relation_dict=relation_dict, 
patience=args.patience,
num_hops=args.hops,
relation_path=relation_embedding_path,
lr=args.lr,
ls=args.ls,
entdrop=args.entdrop,
reldrop=args.reldrop,
scoredrop = args.scoredrop,
model_name=args.model,
decay=args.decay,
w_matrix=w_matrix,
kg_type=kg_type)
