import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm
import re

from utils.misc import init_vocab
from transformers import *
from sparql_evaluator import get_n_hop

mid2ent_file = '/home/ljx/entity_list_file_freebase_complete_all_mention'
kb_vocab_file = '/home/ljx/new_cache_server32_0411/GrailQA_data/bart-base_processed/new_kb_vocab.json'


mid2name = {}
with open(mid2ent_file, 'r') as f:
    for line in f.readlines():
        tmp = line.split('\t')
        mid = tmp[0]
        name = tmp[1]
        mid2name[mid] = name

def get_vocab_from_sparql(sp):
    sp = sp.split('\n')[1:]
    sp = ' '.join(sp)
    sp = sp.split()

    out = []
    for part in sp:
        part = part.strip()
        if part.startswith(":") and not part.startswith(":m.") and not part.startswith(":g."):
            out.append(part[1:])
    #print(' '.join(sp))
    #print(out)
    #input()
    return out

def get_vocab(dataset, vocab, test = False):
    new_vocab = []
    for item in tqdm(dataset):
        qid = item['qid']
        question = item['question']
        #if not test:
        sparql = item['sparql_query']
        # get vocab
        
        #print(question)
        for v in get_vocab_from_sparql(sparql):
            if v not in new_vocab:
                new_vocab.append(v)
    
    #new_vocab = list(set(new_vocab))
    print("New vocab num: %s" % len(new_vocab))
    
    return new_vocab

def main():
    input_dir = '/home/ljx/new_cache_server32_0411/GrailQA_data'
    output_dir = os.path.join(input_dir, 'bart-base_processed')

    print('Load questions')
    train_set = json.load(open(os.path.join(input_dir, 'grailqa_v1.0_train.json')))
    val_set = json.load(open(os.path.join(input_dir, 'grailqa_v1.0_dev.json')))

    all_kb_vocab = []
    for name, dataset in zip(('train', 'val'), (train_set, val_set)):
        print('Get vocab from {} set'.format(name))
        kb_vocab = get_vocab(dataset, name=='test')
        all_kb_vocab.extend(kb_vocab)
    
    all_kb_vocab = list(set(all_kb_vocab))
    vocab_file = os.path.join(output_dir, 'new_kb_vocab_2.json')
    
    print("Total vocab num: %s" % len(all_kb_vocab))
    print("Dump new vocab to %s" % vocab_file)
    with open(vocab_file, 'w') as f:
        json.dump(all_kb_vocab, f)

    
if __name__ == '__main__':
    main()