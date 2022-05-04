# %%
import networkx as nx
from tqdm import tqdm
# Name of data file changed to train.del (WebQSP) from train.txt (refers to MetaQA)
# Train.del contains the KG edges and nodes with ids representing both
f = open('../../data/fbwq_full_new/train.del', 'r')
triples = []
for line in f:
    line = line.strip().split('\t')
    triples.append(line)

# %%
triples[0:5]

# %%
G = nx.Graph()
for t in tqdm(triples):
    e1 = t[0]
    e2 = t[2]
    G.add_node(e1)
    G.add_node(e2)
    G.add_edge(e1, e2)


# %%
G['1594']

# %%
# triples_dict stores all relations between two entities
from collections import defaultdict
triples_dict = defaultdict(set)
for t in tqdm(triples):
    pair = (t[0], t[2])
    triples_dict[pair].add(t[1])

# %%
len(triples_dict)

# %%
# triples_dict stores all relations between two entities
def getRelationsFromKG(head, tail):
    return triples_dict[(head, tail)]


# %%
# Two entities may not be directly connected
# And are connected through other entities
# GetRelationsInPath uses the shortest path and gets relations along the path
# Relations are stored in a SET
def getRelationsInPath(G, e1, e2):
    path = nx.shortest_path(G, e1, e2)
    relations = []
    if len(path) < 2:
        return []
    for i in range(len(path) - 1):
        head = path[i]
        tail = path[i+1]
        rels = list(getRelationsFromKG(head, tail))
        relations.extend(rels)
    return set(relations)

# %%
# !pip install transformers==2.1.1 sentencepiece

# %%
# copied the class pruning_dataloader.py in the notebook
import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
# import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from transformers import RobertaModel, RobertaTokenizer



class DatasetPruning(Dataset):
    def __init__(self, data, rel2idx, idx2rel):
        self.data = data
        self.rel2idx = rel2idx
        self.idx2rel = idx2rel
        self.tokenizer_class = RobertaTokenizer
        self.pretrained_weights = 'roberta-base'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights, cache_dir='.')

    def __len__(self):
        return len(self.data)

    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.rel2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def tokenize_question(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 64)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            # 1 means padding token
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)
    
    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[0]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        rel_ids = data_point[1]
        rel_onehot = self.toOneHot(rel_ids)
        return question_tokenized, attention_mask, rel_onehot


def _collate_fn(batch):
    question_tokenized = batch[0]
    attention_mask = batch[1]
    rel_onehot = batch[2]
    print(len(batch))
    question_tokenized = torch.stack(question_tokenized, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    return question_tokenized, attention_mask, rel_onehot

class DataLoaderPruning(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderPruning, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


# %%
# copied the class pruning_model.py in the notebook
import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformers import RobertaModel

class PruningModel(nn.Module):

    def __init__(self, rel2idx, idx2rel, ls):
        super(PruningModel, self).__init__()
        self.label_smoothing = ls
        self.rel2idx = rel2idx
        self.idx2rel = idx2rel

        self.roberta_pretrained_weights = 'roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)

        self.roberta_dim = 768
        self.mid1 = 512
        self.mid2 = 512
        self.mid3 = 256
        self.mid4 = 256
        self.fcnn_dropout = torch.nn.Dropout(0.1)
        # self.lin1 = nn.Linear(self.roberta_dim, self.mid1)
        # self.lin2 = nn.Linear(self.mid1, self.mid2)
        # self.lin3 = nn.Linear(self.mid2, self.mid3)
        # self.lin4 = nn.Linear(self.mid3, self.mid4)
        # self.hidden2rel = nn.Linear(self.mid4, len(self.rel2idx))
        self.hidden2rel = nn.Linear(self.roberta_dim, len(self.rel2idx))
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)        
        
    def applyNonLinear(self, outputs):
        # outputs = self.fcnn_dropout(self.lin1(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin2(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin3(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin4(outputs))
        # outputs = F.relu(outputs)
        outputs = self.hidden2rel(outputs)
        # outputs = self.hidden2rel_base(outputs)
        return outputs

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding
    
    def forward(self, question_tokenized, attention_mask, rel_one_hot):
        question_embedding = self.getQuestionEmbedding(question_tokenized, attention_mask)
        prediction = self.applyNonLinear(question_embedding)
        prediction = torch.sigmoid(prediction)
        actual = rel_one_hot
        if self.label_smoothing:
            actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1)) 
        loss = self.loss(prediction, actual)
        return loss
        

    def get_score_ranked(self, question_tokenized, attention_mask):
        question_embedding = self.getQuestionEmbedding(question_tokenized.unsqueeze(0), attention_mask.unsqueeze(0))
        prediction = self.applyNonLinear(question_embedding)
        prediction = torch.sigmoid(prediction).squeeze()
        # top2 = torch.topk(scores, k=2, largest=True, sorted=True)
        # return top2
        return prediction
        

# %%
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
import networkx as nx
from collections import defaultdict
# from pruning_model import PruningModel
# from pruning_dataloader import DatasetPruning, DataLoaderPruning

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# pretrained model provided by the author is trained on Relations All
# while the Knowledge Graph is using relation ids from a pruned list
# Since both the relation ids are used and there is a mismatch between the IDs
# Matching happens after looking up RELATIONS from respective IDs

f = open('../../data/relations_all.dict', 'r')
# f = open('../../data/fbwq_full_new/relation_ids.del', 'r')
rel2idx = {}
idx2rel = {}
for line in f:
    line = line.strip().split('\t')
    id = int(line[1])
    rel = line[0]
    # id = int(line[0])
    # rel = line[1]
    rel2idx[rel] = id
    idx2rel[id] = rel
f.close()

def process_data_file(fname, rel2idx, idx2rel):
    f = open(fname, 'r')
    data = []
    for line in f:
        line = line.strip().split('\t')
        question = line[0].strip()
        #TODO only work for webqsp. to remove entity from metaqa, use something else
        #remove entity from question
        question = question.split('[')[0]
        rel_list = line[1].split('|')
        rel_id_list = []
        for rel in rel_list:
            rel_id_list.append(rel2idx[rel])
        data.append((question, rel_id_list, line[0].strip()))
    return data



# %%
model = PruningModel(rel2idx, idx2rel, 0.0)
checkpoint_file = "../../pretrained_models/relation_matching_models/webqsp.pt"
# model.load(checkpoint_file)
st_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
print(st_dict.keys())
model.load_state_dict(st_dict)


# %%

data = process_data_file('../../data/fbwq_full_new/pruning_train.txt', rel2idx, idx2rel)
dataset = DatasetPruning(data=data, rel2idx = rel2idx, idx2rel = idx2rel)
print('Done')

# %%
dataset.data[0]


# %%
# Knowledge Graph -> get nodes at 1 hop, 2 hop, 3 hop distance

# Sample Question 'q'
# "what is the name of justin bieber brother [m.06w2sn5]	people.sibling_relationship.sibling|people.person.sibling_s"
# For a given question, return the entity name "m.06w2sn5" which refers to "Justin Bieber" in above question
# Entity ids are numeric ids
def getHead(q):
    question = q.split('[')
    question_1 = question[0]
    question_2 = question[1].split(']')
    head = question_2[0].strip()
    return head

def get2hop(graph, entity):
    l1 = graph[entity]
    ans = []
    ans += l1
    for item in l1:
        ans += graph[item]
    ans = set(ans)
    if entity in ans:
        ans.remove(entity)
    return ans

def get3hop(graph, entity):
    l1 = graph[entity]
    ans = []
    ans += l1
    for item in l1:
        ans += graph[item]
    ans2 = []
    ans2 += ans
    for item in ans:
        ans2 += graph[item]
    ans2 = set(ans2)
    if entity in ans2:
        ans2.remove(entity)
    return ans2

def get1hop(graph, entity):
    l1 = graph[entity]
    ans = []
    ans += l1
    ans = set(ans)
    if entity in ans:
        ans.remove(entity)
    return ans


def getnhop(graph, entity, hops=1):
    if hops == 1:
        return get1hop(graph, entity)
    elif hops == 2:
        return get2hop(graph, entity)
    else:
        return get3hop(graph, entity)



# %%
def getAllRelations(head, tail):
    global G
    global triples_dict
    try:
        shortest_length = nx.shortest_path_length(G, head, tail)
    except:
        shortest_length = 0
    if shortest_length == 0:
        return set()
    if shortest_length == 1:
        return triples_dict[(head, tail)]
    elif shortest_length == 2:
        paths = [nx.shortest_path(G, head, tail)]
        relations = set()
        for p in paths:
            rels1 = triples_dict[(p[0], p[1])]
            rels2 = triples_dict[(p[1], p[2])]
            relations = relations.union(rels1)
            relations = relations.union(rels2)
        return relations
    else:
      # if shortest path is more than length 2, empty set is returned
      # that is only <2 hop paths are considered, remaining paths are ignored
        return set()
    

# %%
# not used in the code anywhere
def removeHead(question):
    question = question = question.split('[')[0]
    return question

# %%
# subset of questions for faster testing/tuning
# num_for_testing = 100
import pickle
cws = pickle.load(open('webqsp_scores_full_kg.pkl', 'rb'))
num_for_testing = len(cws)
print(cws[0])
len(cws)


# %%
# length of scores and candidates is same
len(cws[0]["scores"])
len(cws[0]["candidates"])

# %%
import pandas as pd
entity2id = pd.read_csv('../../data/fbwq_full_new/entity_ids.del', '\t', header = None)
print(len(entity2id))
print(entity2id[1:10])

# %%
eid = entity2id.to_numpy()
print(eid[0:10])
id = np.where(eid == 'm.02nzb8')
print(id[0])

# %%
# pretrained model provided by the author is trained on Relations All (relations_all.dict)
# while the Knowledge Graph is using relation ids from a pruned list (relation_ids.del)
# Since both the relation ids are used and there is a mismatch between the IDs
# Matching happens after looking up RELATIONS from respective IDs

f = open('../../data/fbwq_full_new/relation_ids.del', 'r')
mmrel2idx = {}
mmidx2rel = {}
for line in f:
    line = line.strip().split('\t')
    id = int(line[0])
    rel = line[1]
    mmrel2idx[rel] = id
    mmidx2rel[id] = rel
f.close()


# %%
# num_for_testing = 500

# %%
# Algorithm mentioned in paper, sec 4.4.1
# slower than the previous one but does not have neighbourhood restriction
num_correct = 0
for q in tqdm(cws[:num_for_testing]):
    question = q['question']
    question_nohead = question
    answers = q['answers']
    candidates = q['candidates']
    candidates_scores = q['scores']
    head = q['head']
    question_tokenized, attention_mask = dataset.tokenize_question(question)
    scores = model.get_score_ranked(question_tokenized=question_tokenized, attention_mask=attention_mask)
    pruning_rels_scores, pruning_rels_torch = torch.topk(scores, 2)
    pruning_rels = set()
    pruning_rels_threshold = 0.5 # threshold to consider as written in sec 4.4.1
    for s, p in zip(pruning_rels_scores, pruning_rels_torch):
        if s > pruning_rels_threshold:
            pruning_rels.add(idx2rel[p.item()])
            # pruning_rels.add(p.item())
    gamma = 1.0
    max_score = 0.0
    max_score_relscore = 0.0
    max_score_answer = ""
    head1 = np.where(eid == head)[0]
    head1 = np.array2string(head1).strip('[]')
    for score, c in zip(candidates_scores, candidates):
        c1 = np.where(eid == c)[0]
        c1 = np.array2string(c1).strip('[]')
        
        actual_relids = getAllRelations(head1, c1)
        actual_rels= set()
        for id1 in actual_relids:
          actual_rels.add(mmidx2rel[int(id1)])

        relscore = len(actual_rels.intersection(pruning_rels))
        totalscore = score + gamma*relscore
        if totalscore > max_score:
            max_score = totalscore
            max_score_relscore = relscore
            max_score_answer = c
    is_correct = False
    if max_score_answer in answers:
        num_correct += 1
        is_correct = True    

# %%
print('Accuracy is', num_correct/num_for_testing)

# %% [markdown]
# Accuracy on first 100 instances = 61%,
# Accuracy on next 500 instances = 58.6%,
# Accuracy on next 500 instances = 58.6% (to repeat 600 to 1100)

# %%
# this is an alternative type of relation matching using neighbourhood
# this was used in ablation, but params are not matching (since this notebook was
# used for experimentation)
# this is much faster than the algorithm mentioned in paper
num_correct = 0
numEmpty  = 0
for q in tqdm(cws[:num_for_testing]):
    question = q['question']
    question_nohead = question
    answers = q['answers']
    # print(answers)
    candidates = q['candidates']
    head = q['head']

    # the below model returns top RELATIONS associated with the question
    # which will be matched with the relations that candidates have with the head
    question_tokenized, attention_mask = dataset.tokenize_question(question)
    scores = model.get_score_ranked(question_tokenized=question_tokenized, attention_mask=attention_mask)
    pruning_rels_scores, pruning_rels_torch = torch.topk(scores, 5)
    # print(pruning_rels_scores)
    pruning_rels = set()
    pruning_rels_threshold = 0.5
    for s, p in zip(pruning_rels_scores, pruning_rels_torch):
        if s > pruning_rels_threshold:
            pruning_rels.add(idx2rel[p.item()])
            # pruning_rels.add(p.item())

    # print("pruning rels = ", pruning_rels)

    my_answer = ""
    head1 = np.where(eid == head)[0]
    head1 = np.array2string(head1).strip('[]')
    head_nbhood = get2hop(G, head1)
    # print(head_nbhood)
#     max_intersection = 0
    for c in candidates:
#         candidate_rels = getAllRelations(head, c)
        c1 = np.where(eid == c)[0]
        c1 = np.array2string(c1).strip('[]')
        if c1 in head_nbhood:
            # print(c1)
            candidate_relids = getAllRelations(head1, c1)
            candidate_rels= set()
            for id1 in candidate_relids:
              candidate_rels.add(mmidx2rel[int(id1)])

            # print(c, candidate_relids)
            intersection = pruning_rels.intersection(candidate_rels)
            # intersection = pruning_rels.intersection(candidate_relids)
            if len(intersection) > 0:
                # print("intersection", intersection)
                # since candidates are sorted in the order of scores, break whenver the relationship match is found
                my_answer = c
                break
    if my_answer == "":
        my_answer = candidates[0]
    # print(my_answer)
    # print("\n")
    if my_answer in answers:
        num_correct += 1
print('Accuracy is', num_correct/num_for_testing)

# %%


# %%


# %%
# following code is just for investigating
# i haven't removed it since it might be useful if
# someone wants to explore

qid=2
model.eval()
qe = cws[qid]['qe']
p1, p2 = torch.topk(model.get_score_ranked(qe), 5)
question = cws[qid]['question']
print(question)
# print(idx2rel[pred])
for p in p2:
    print(idx2rel[p.item()])
candidates = cws[qid]['candidates']
print(candidates)
# print(cws[qid]['scores'])
head = getHead(question)
tail = candidates[1]
getAllRelations(head, tail)

# %%
for p in paths:
    print(p)

# %%
e1 = 'm.01_2n'
e2 = 'm.0bvv2dt'
e1 = np.where(eid == e1)[0]
e1 = np.array2string(e1).strip('[]')

e2 = np.where(eid == e2)[0]
e2 = np.array2string(e2).strip('[]')
print(e1, e2)
# getRelationsFromKG(e1, e2)
getRelationsFromKG(e1, '138')

# %%
getRelationsFromKG('m.04j60kh', 'm.02_bcst')

# %%



