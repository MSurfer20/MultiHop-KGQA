import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np


class DatasetMetaQA(Dataset):
    def __init__(self, data, word2ix, relations, entities, entity2idx):
        self.data = data
        self.relations = relations
        self.entities = entities
        self.word_to_ix = {}
        self.entity2idx = entity2idx
        self.word_to_ix = word2ix
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.index_array = list(self.entities.keys())


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_ids = [self.word_to_ix[word] for word in question_text.split()]
        head_id = self.entity2idx[data_point[0].strip()]
        tail_ids = []
        for tail_name in data_point[2]:
            tail_name = tail_name.strip()
            tail_ids.append(self.entity2idx[tail_name])
        indices = torch.LongTensor(tail_ids)
        batch_size = len(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return question_ids, head_id, one_hot 

class DataLoaderMetaQA(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderMetaQA, self).__init__(*args, **kwargs)

