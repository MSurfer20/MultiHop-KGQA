import torch
def create_embedding_dics(embedding_dict):
    entity2idx = dict()
    idx2entity = dict()
    embedding_matrix = list()
    i = 0
    for key, entity in embedding_dict.items():
        entity2idx[key.strip()] = i
        idx2entity[i] = key.strip()
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix

def get_vocab(data):
    word_to_ix = dict()
    maxLength = 0
    curr_word_idx=0
    idx2word = dict()
    for d in data:
            sent = d[1]
            for word in sent.split():
                if word in word_to_ix:
                    continue
                idx2word[curr_word_idx] = word
                word_to_ix[word] = curr_word_idx
                curr_word_idx+=1
                    
            if len(sent.split()) > maxLength:
                maxLength = len(sent.split())

    return word_to_ix, idx2word, maxLength

def parse_text_file(text_file, split=False):
    data_file = open(text_file, 'r')
    data_array = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        question = data_line[0].split('[')
        question_1 = question[0]
        question_2 = question[1].split(']')
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1+'NE'+question_2
        ans = data_line[1].split('|')
        data_array.append([head, question.strip(), ans])
    if split==False:
        return data_array
    else:
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            for tail in tails:
                data.append([head, question, tail])
        return data

def data_generator(data, word2ix, entity2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1].strip().split(' ')
        encoded_question = [word2ix[word.strip()] for word in question]
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long),torch.tensor(encoded_question, dtype=torch.long) , ans, torch.tensor(len(encoded_question), dtype=torch.long), data_sample[1]
