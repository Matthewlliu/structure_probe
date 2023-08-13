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
import torch

from utils.misc import init_vocab
from transformers import *
from sparql_evaluator import get_n_hop

mid2ent_file = '/home/ljx/entity_list_file_freebase_complete_all_mention'
#entlink_file = '/home/ljx/GrailQA-main/entity_linking/grailqa_el.json'
#with open(entlink_file, 'r') as f:
#    grailqa_el = json.load(f)
kb_vocab_file = '/home/ljx/new_cache_server32_0411/GrailQA_data/bart-base_processed/new_kb_vocab_2.json'


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
            out.append(part)
    return out

def doc_anonymize(question, sp):
    sp = sp.split('\n')[1:]
    sp = ' '.join(sp)
    sp = sp.split()

    #count = 0
    template = '[%s]'
    ent = {}
    for i in range(len(sp)):
        part = sp[i]
        if part.startswith(":m.") or part.startswith(":g."):
            mid = part[1:]
            name = mid2name[mid]
            ent[template%name] = [mid, name]
            #question = question.replace(name, template % name)
            sp[i] = ':'+template % name
            #count += 1
    return question, ' '.join(sp), ent

def encode_dataset(dataset, vocab, tokenizer, test = False):
    questions = []
    sparqls = []
    new_vocab = []
    for item in tqdm(dataset):
        qid = item['qid']
        question = item['question']
        #if not test:
        sparql = item['sparql_query']
        # get vocab
        
        '''
        for v in get_vocab_from_sparql(sparql):
            if v not in new_vocab:
                new_vocab.append(v)
        '''
        question, sparql, ent = doc_anonymize(question, sparql)
        '''
        for k,v in ent.items():
            one_hop_ent, one_hop_rel = get_n_hop(v[0], 1)
            two_hop_ent, two_hop_rel = get_n_hop(v[0], 2)
            for item in chain(one_hop_ent, one_hop_rel, two_hop_ent, two_hop_rel):
                if item not in new_vocab:
                    new_vocab.append(item)
        '''
        
        questions.append(question)
        sparqls.append(sparql)
    sequences = questions + sparqls
    encoded_inputs = tokenizer(sequences, padding = True)
    #new_vocab = list(set(new_vocab))
    #print("New vocab num: %s" % len(new_vocab))
    print(encoded_inputs.keys())
    print(encoded_inputs['input_ids'][0])
    print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    print(tokenizer.decode(encoded_inputs['input_ids'][-1]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    assert max_seq_length == len(encoded_inputs['input_ids'][-1])
    print(max_seq_length)
    questions = []
    sparqls = []
    entities = []
    answers = []
    for item in tqdm(dataset):
        question = item['question']
        #_ = [vocab['answer_token_to_idx'][w] for w in item['choices']]
        #choices.append(_)
        #if not test:
        sparql = item['sparql_query']

        question, sparql, ent = doc_anonymize(question, sparql)
        questions.append(question)
        sparqls.append(sparql)
        entities.append(ent)
        #answers.append(vocab['answer_token_to_idx'].get(item['answer']))
        tmp = []
        for a in item['answer']:
            tmp.append(vocab['answer_token_to_idx'].get(a['answer_argument']))
        answers.append(tmp)

    input_ids = tokenizer.batch_encode_plus(questions, max_length = max_seq_length, pad_to_max_length = True, truncation = True)
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)
    if not test:
        target_ids = tokenizer.batch_encode_plus(sparqls, max_length = max_seq_length, pad_to_max_length = True, truncation = True)
        target_ids = np.array(target_ids['input_ids'], dtype = np.int32)
    else:
        target_ids = np.array([], dtype = np.int32)
    #choices = np.array(choices, dtype = np.int32)
    #answers = np.array(answers, dtype = np.int32)
    return source_ids, source_mask, target_ids, entities, answers, new_vocab

def split_vocab_name(v):
    out = []
    v = v.split('.')
    for vv in v:
        out += vv.split('_')
    return out
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    args = parser.parse_args()

    print('Build kb vocabulary')
    vocab = {
        'answer_token_to_idx': {}
    }
    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'grailqa_v1.0_train.json')))
    val_set = json.load(open(os.path.join(args.input_dir, 'grailqa_v1.0_dev.json')))
    #test_set = json.load(open(os.path.join(args.input_dir, 'grailqa_v1.0_test_public.json')))
    #max_ans_num = 0
    for question in tqdm(chain(train_set, val_set)):
        #if len(question['answer'])>max_ans_num:
        #    max_ans_num = len(question['answer'])
        for a in question['answer']:
            if not a['answer_argument'] in vocab['answer_token_to_idx']:
                vocab['answer_token_to_idx'][a['answer_argument']] = len(vocab['answer_token_to_idx'])

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok = True)
    fn = os.path.join(args.output_dir, 'vocab.json')
    print('Dump vocab to {}'.format(fn))
    with open(fn, 'w') as f:
        json.dump(vocab, f, indent=2)
    for k in vocab:
        print('{}:{}'.format(k, len(vocab[k])))
    #model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    with open(kb_vocab_file, 'r') as f:
        kb_new_vocab = json.load(f)
    for token in tqdm(kb_new_vocab):
        #token = token[1:]
        #if len(token)>100 or '.' not in token:
        #    continue
        tokenizer.add_tokens(token, verbose=False)
    #if len(kb_new_vocab) > 0:
    #    model.resize_token_embeddings(len(tokenizer))
    '''
    # Assigning embeddings to kv_vocab
    target_tokens = kb_new_vocab
    origin_tokens = [ split_vocab_name(v) for v in kb_new_vocab ]

    for o, t in zip(origin_tokens, target_tokens):
        # o_token_ids = [tokenizer.convert_tokens_to_ids([item])[0] for item in o]
        if isinstance(o, list):
            if len(o) == 1:
                o_tokens = tokenizer.tokenize(o[0])
                o_token_ids = tokenizer.convert_tokens_to_ids(o_tokens)
                # print(o, tokenizer.convert_tokens_to_ids(o))
                if isinstance(o_token_ids, list):
                    o_token_ids = [o_token_ids]
                else:
                    o_token_ids = [[o_token_ids]]
            else:
                o_token_ids = []
                for item in o:
                    # print("multi-", item)
                    item_tokens = tokenizer.tokenize(item)
                    item_token_ids = tokenizer.convert_tokens_to_ids(item_tokens)
                    # print(item_token_ids)
                    if isinstance(item_token_ids, list):
                        o_token_ids.append(item_token_ids)
                    else:
                        o_token_ids.append([item_token_ids])
            o_counts = len(o)

            t_token_id = tokenizer.convert_tokens_to_ids([t])[0]
            # print(o)
            # print(o_token_ids)

            with torch.no_grad():
                print(f"Assign [[%s]] to [[%s]]" % (o, t))
                if o_counts == 1:
                    encoder_token_embeds = 0
                    decoder_token_embeds = 0
                    for o_token_id in o_token_ids[0]:
                        encoder_token_embeds += model.model.encoder.embed_tokens.weight[o_token_id].detach().clone()
                        decoder_token_embeds += model.model.decoder.embed_tokens.weight[o_token_id].detach().clone()
                    model.model.encoder.embed_tokens.weight[t_token_id] = encoder_token_embeds
                    model.model.decoder.embed_tokens.weight[t_token_id] = decoder_token_embeds
                else:
                    encoder_token_embeds = 0
                    decoder_token_embeds = 0
                    for item in o_token_ids:
                        for o_token_id in item:
                            encoder_token_embeds += model.model.encoder.embed_tokens.weight[o_token_id].detach().clone()
                            decoder_token_embeds += model.model.decoder.embed_tokens.weight[o_token_id].detach().clone()
                    encoder_token_embeds = encoder_token_embeds / o_counts
                    decoder_token_embeds = decoder_token_embeds / o_counts
                    model.model.encoder.embed_tokens.weight[t_token_id] = encoder_token_embeds
                    model.model.decoder.embed_tokens.weight[t_token_id] = decoder_token_embeds
    '''
    #model.save_pretrained(os.path.join(args.output_dir, 'model'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    
    #all_kb_vocab = []
    for name, dataset in zip(('train', 'val'), (train_set, val_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(dataset, vocab, tokenizer, name=='test')
        kb_vocab = outputs[5]
        #all_kb_vocab.extend(kb_vocab)
        outputs = outputs[:5]
        assert len(outputs) == 5
        print('shape of input_ids of questions, attention_mask of questions, input_ids of sparqls, choices and answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                if isinstance(o, list):
                    print(len(o))
                else:
                    print(o.shape)
                pickle.dump(o, f)
    '''
    all_kb_vocab = list(set(all_kb_vocab))
    vocab_file = os.path.join(args.output_dir, 'new_kb_vocab.json')
    print(all_kb_vocab)
    print("Dump new vocab to %s" % vocab_file)
    with open(vocab_file, 'w') as f:
        json.dump(all_kb_vocab, f)
    '''
    
if __name__ == '__main__':
    main()