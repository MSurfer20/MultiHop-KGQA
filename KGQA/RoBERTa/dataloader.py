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
from transformers import *
from helpers import *

class DatasetWebQSP(Dataset):
    def __init__(self, data, entities, entity2idx, transformer_name, kg_model):
        self.pos_dict = defaultdict(list)
        self.data = data
        self.neg_dict = defaultdict(list)
        self.entities = entities
        self.transformer_name = transformer_name
        self.max_length = 64
        self.entity2idx = entity2idx
        self.kg_model = kg_model
        self.index_array = list(self.entities.keys())
        self.pre_trained_model_name = get_pretrained_model_name(transformer_name)
        self.tokenizer = None
        if self.transformer_name == 'SentenceTransformer':
            self.tokenizer = AutoTokenizer.from_pretrained(self.pre_trained_model_name)
        elif self.transformer_name == 'RoBERTa':
            self.tokenizer = RobertaTokenizer.from_pretrained(self.pre_trained_model_name)
        elif self.transformer_name == 'ALBERT':
            self.tokenizer = AlbertTokenizer.from_pretrained(self.pre_trained_model_name)
        else:
            print('Incorrect transformer specified:', self.transformer_name)
            exit(0)
        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        head_id = self.entity2idx[data_point[0].strip()]
        tail_ids = []
        for tail_name in data_point[2]:
            tail_name = tail_name.strip()
            if tail_name in self.entity2idx:
                tail_ids.append(self.entity2idx[tail_name])
        indices = torch.LongTensor(tail_ids)
        batch_size = len(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return question_tokenized, attention_mask, head_id, one_hot 

    def tokenize_question(self, question):
        if self.transformer_name != "SentenceTransformer":
            question = f"<s>{question}</s>"
            question_tokenized = self.tokenizer.tokenize(question)

            num_to_add = self.max_length_len - len(question_tokenized)
            for i in range(num_to_add):
                question_tokenized.append('<pad>')

            question_tokenized = torch.tensor(self.tokenizer.encode(
                                    question_tokenized, # Question to encode
                                    add_special_tokens = False # Add '[CLS]' and '[SEP]', as per original paper
                                    ))

            attention_mask = list()
            for q in question_tokenized:
                # 1 means padding token
                if q == 1:
                    attention_mask.append(0)
                else:
                    attention_mask.append(1)

            return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)
        else:

            encoded_que = self.tokenizer.encode_plus(question, padding='max_length', max_length=self.max_length, return_tensors='pt')
            return encoded_que['input_ids'][0], encoded_que['attention_mask'][0]


class DataLoaderWebQSP(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderWebQSP, self).__init__(*args, **kwargs)

