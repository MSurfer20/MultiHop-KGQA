import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
from dataloader import DatasetWebQSP, DataLoaderWebQSP
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import networkx as nx
import time
import sys
import pandas as pd
sys.path.append("../..") # Adds higher directory to python modules path.
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from torch.nn.utils.rnn import pad_sequence
from text_process import create_embedding_dics, data_generator, process_text_file

parser = argparse.ArgumentParser()


parser.add_argument('--hops', type=str, default='1')
parser.add_argument('--load_from', type=str, default='')
parser.add_argument('--ls', type=float, default=0.0)
parser.add_argument('--decay', type=float, default=1.0)
parser.add_argument('--validate_every', type=int, default=5)
parser.add_argument('--model', type=str, default='ComplEx')
parser.add_argument('--mode', type=str, default='eval')
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--outfile', type=str, default='best_score_model')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--entdrop', type=float, default=0.0)
parser.add_argument('--reldrop', type=float, default=0.0)
parser.add_argument('--scoredrop', type=float, default=0.0)
parser.add_argument('--l3_reg', type=float, default=0.0)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nb_epochs', type=int, default=2)
parser.add_argument('--relation_dim', type=int, default=30)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--freeze', type=bool, default=True)
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--que_embedding_model', type=str, default='RoBERTa')
parser.add_argument('--do_batch_norm', type=bool, default=True)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
args = parser.parse_args()

def inTopk(scores, ans, k):
    result = False
    topk = torch.topk(scores, k)[1]
    for x in topk:
        if x in ans:
            result = True
    return result

def test(data_path, device, model, dataloader, entity2idx, model_name, return_hits_at_k):
    model.eval()
    data = process_text_file(data_path)
    idx2entity = dict()
    for key, value in entity2idx.items():
        idx2entity[value] = key
    answers = list()
    data_gen = data_generator(data=data, dataloader=dataloader, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    num_incorrect = 0
    incorrect_rank_sum = 0
    not_in_top_50_count = 0

    scores_list = []
    hit_at_1 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    candidates_with_scores = []
    writeCandidatesToFile = True
    for i in tqdm(range(len(data))):
        d = next(data_gen)
        head = d[0].to(device)
        question_tokenized = d[1].to(device)
        attention_mask = d[2].to(device)
        ans = d[3]
        tail_test = torch.tensor(ans, dtype=torch.long).to(device)
        scores = model.get_score_ranked(head=head, question_tokenized=question_tokenized, attention_mask=attention_mask)[0]
        # following 2 lines for no neighbourhood check
        mask = torch.zeros(len(entity2idx)).to(device)
        mask[head] = 1
        #reduce scores of all non-candidates
        new_scores = scores - (mask*99999)
        pred_ans = torch.argmax(new_scores).item()
        if pred_ans == head.item():
            print('Head and answer same')
            print(torch.max(new_scores))
            print(torch.min(new_scores))
        
        if writeCandidatesToFile:
            entry = {}
            entry['question'] = d[-1]
            head_text = idx2entity[head.item()]
            entry['head'] = head_text
            s, c =  torch.topk(new_scores, 200)
            s = s.cpu().detach().numpy()
            c = c.cpu().detach().numpy()
            cands = []
            for cand in c:
                cands.append(idx2entity[cand])
            entry['scores'] = s
            entry['candidates'] = cands
            correct_ans = []
            for a in ans:
                correct_ans.append(idx2entity[a])
            entry['answers'] = correct_ans
            candidates_with_scores.append(entry)


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
        
    if writeCandidatesToFile:
        pickle.dump(candidates_with_scores, open('webqsp_scores_finetune.pkl', 'wb'))
        print('wrote candidate file (for future answer processing)')
    accuracy = total_correct/len(data)

    if return_hits_at_k:
        return answers, accuracy, (hit_at_1/len(data)), (hit_at_5/len(data)), (hit_at_10/len(data))
    else:
        return answers, accuracy

def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('Wrote to ', fname)
    return

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()

def getEntityEmbeddings(model_name, kge_model, hops):
    e = {}
    model_dir = f"../../pretrained_models/embeddings/{model_name}"
    entity_dict = f"{model_dir}_fbwq_full/entity_ids.del"

    if 'half' in hops:
        entity_dict = f"{model_dir}_fbwq_half/entity_ids.del"
        print('Loading half entity_ids.del')
    embedder = kge_model._entity_embedder
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = embedder._embeddings(torch.LongTensor([ent_id]))[0]
    f.close()
    return e

def get_chkpt_path(model_name, que_embedding_model, outfile):
    return f"../../checkpoints/WebQSP/{model_name}_{que_embedding_model}_{outfile}/best_score_model.pt"

def perform_experiment(data_path, mode, batch_size, num_workers, nb_epochs, embedding_dim, hidden_dim, relation_dim, gpu ,patience, freeze, validate_every, hops, lr, entdrop, reldrop, scoredrop, l3_reg, model_name, decay, ls, load_from, outfile, do_batch_norm, que_embedding_model, valid_data_path=None, test_data_path=None):
    webqsp_checkpoint_folder = f"../../checkpoints/WebQSP/{model_name}_{que_embedding_model}_{outfile}/"
    if not os.path.exists(webqsp_checkpoint_folder):
        os.makedirs(webqsp_checkpoint_folder)
    
    print('Loading entities and relations')
    kg_type = 'full'
    if 'half' in hops:
        kg_type = 'half'
    
    checkpoint_file = f"../../pretrained_models/embeddings/{model_name}_fbwq_{kg_type}/checkpoint_best.pt"

    print('Loading kg embeddings from', checkpoint_file)
    kge_checkpoint = load_checkpoint(checkpoint_file)
    kge_model = KgeModel.create_from(kge_checkpoint)
    kge_model.eval()
    e = getEntityEmbeddings(model_name, kge_model, hops)

    print('Loaded entities and relations')

    entity2idx, idx2entity, embedding_matrix = create_embedding_dics(e)

    # word2ix,idx2word, max_len = get_vocab(data)
    # hops = str(num_hops)
    device = torch.device('cuda')
    model = RelationExtractor(embedding_dim=embedding_dim, num_entities = len(idx2entity), relation_dim=relation_dim, pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop = entdrop, reldrop = reldrop, scoredrop = scoredrop, l3_reg = l3_reg, model = model_name, que_embedding_model=que_embedding_model, ls = ls, do_batch_norm=do_batch_norm)

    # time.sleep(10)
    if mode=='train':
        data = process_text_file(data_path)
        dataset = DatasetWebQSP(data, e, entity2idx, que_embedding_model, model_name)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        if load_from != '':
            # model.load_state_dict(torch.load("checkpoints/roberta_finetune/" + load_from + ".pt"))
            #fname = f"checkpoints/{que_embedding_model}_finetune/{load_from}.pt"
            fname = f"../../checkpoints/WebQSP/{model_name}_{que_embedding_model}_{outfile}/{load_from}.pt"
            model.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, decay)
        optimizer.zero_grad()
        best_score = -float("inf")
        best_model = model.state_dict()
        no_update = 0
        for epoch in range(nb_epochs):
            phases = []
            for i in range(validate_every):
                phases.append('train')
            phases.append('valid')
            for phase in phases:
                if phase == 'train':
                    model.train()
                    # model.apply(set_bn_eval)
                    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                    running_loss = 0
                    for i_batch, a in enumerate(loader):
                        model.zero_grad()
                        question_tokenized = a[0].to(device)
                        attention_mask = a[1].to(device)
                        positive_head = a[2].to(device)
                        positive_tail = a[3].to(device)    
                        loss = model(question_tokenized=question_tokenized, attention_mask=attention_mask, p_head=positive_head, p_tail=positive_tail)
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
                    answers, score = test(model=model, data_path= valid_data_path, entity2idx=entity2idx, dataloader=dataset, device=device, model_name=model_name, return_hits_at_k=False)
                    if score > best_score + eps:
                        best_score = score
                        no_update = 0
                        best_model = model.state_dict()
                        print(hops + " hop Validation accuracy (no relation scoring) increased from previous epoch", score)
                        writeToFile(answers, f'results/{model_name}_{que_embedding_model}_{outfile}.txt')
                        torch.save(best_model, get_chkpt_path(model_name, que_embedding_model, outfile))
                    elif (score < best_score + eps) and (no_update < patience):
                        no_update +=1
                        print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, patience-no_update))
                    elif no_update == patience:
                        print("Model has exceed patience. Saving best model and exiting")
                        torch.save(best_model, get_chkpt_path(model_name, que_embedding_model, outfile))
                        exit(0)
                    if epoch == nb_epochs-1:
                        print("Final Epoch has reached. Stoping and saving model.")
                        torch.save(best_model, get_chkpt_path(model_name, que_embedding_model, outfile))
                        exit()
                    # torch.save(model.state_dict(), "checkpoints/roberta_finetune/"+str(epoch)+".pt")
                    # torch.save(model.state_dict(), "checkpoints/roberta_finetune/x.pt")   
    
    elif mode=='test':
        data = process_text_file(test_data_path)
        dataset = DatasetWebQSP(data, e, entity2idx, que_embedding_model, model_name)
        model_chkpt_file_path = get_chkpt_path(model_name, que_embedding_model, outfile)
        model.load_state_dict(torch.load(model_chkpt_file_path, map_location=lambda storage, loc: storage))
        model.to(device)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        answers, accuracy, hits_at_1, hits_at_5, hits_at_10 = test(model=model, data_path= test_data_path, entity2idx=entity2idx, dataloader=dataset, device=device, model_name=model_name, return_hits_at_k=True)

        d = {
            'KG-Model': model_name,
            'KG-Type': kg_type,
            'Que-Embedding-Model': que_embedding_model,
            'Accuracy': [accuracy], 
            'Hits@1': [hits_at_1],
            'Hits@5': [hits_at_5],
            'Hits@10': [hits_at_10]
            }
        df = pd.DataFrame(data=d)
        df.to_csv(f"final_results.csv", mode='a', index=False, header=False)
                
hops = args.hops

model_name = args.model

if 'webqsp' in hops:
    data_path = '../../data/QA_data/WebQuestionsSP/qa_train_webqsp.txt'
    valid_data_path = '../../data/QA_data/WebQuestionsSP/qa_test_webqsp.txt'
    test_data_path = '../../data/QA_data/WebQuestionsSP/qa_test_webqsp.txt'



perform_experiment(
    data_path=data_path, 
    mode=args.mode,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    nb_epochs=args.nb_epochs, 
    embedding_dim=args.embedding_dim, 
    hidden_dim=args.hidden_dim, 
    relation_dim=args.relation_dim, 
    valid_data_path=valid_data_path,
    test_data_path=test_data_path,
    patience=args.patience,
    validate_every=args.validate_every,
    freeze=args.freeze,
    hops=args.hops,
    lr=args.lr,
    entdrop=args.entdrop,
    reldrop=args.reldrop,
    scoredrop = args.scoredrop,
    l3_reg = args.l3_reg,
    model_name=args.model,
    decay=args.decay,
    ls=args.ls,
    load_from=args.load_from,
    outfile=args.outfile,
    do_batch_norm=args.do_batch_norm,
    que_embedding_model=args.que_embedding_model
)
