from SPARQLWrapper import SPARQLWrapper, JSON
import os
import json
import re
from tqdm import tqdm
from bm25 import BM25_Model
import numpy as np

PREFIX_DICT = {
    "rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
    ":": "http://rdf.freebase.com/ns/"
}
PREFIX = 'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> '

ent2hop_file = '/home/ljx/structure_probe/KQAPro_Baselines-master/grailqa_dev_ent2hop_info.json'
with open(ent2hop_file, 'r') as f:
    ent2hop = json.load(f)

def get_sparql_ans(query):
    """
        return: List of answers
    """
    sparql = SPARQLWrapper('http://localhost:23621/sparql')
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    out = []
    try:
        results = sparql.query().convert()  # json,type为dict
        results = results['results']
        for r in results['bindings']:
            if r['value']['value'].startswith(PREFIX_DICT['rdf:']):
                tmp = r['value']['value'].replace(PREFIX_DICT['rdf:'], '')
            elif r['value']['value'].startswith(PREFIX_DICT['rdfs:']):
                tmp = r['value']['value'].replace(PREFIX_DICT['rdfs:'], '')
            elif r['value']['value'].startswith(PREFIX_DICT[':']):
                tmp = r['value']['value'].replace(PREFIX_DICT[':'], '')
            out.append(tmp)
    except:
        out = []
    return out

def preprocess(query, entities):
    query = PREFIX + query 
    for k, v in entities.items(): # change [E*] to mid
        query = query.replace(k, v[0])

    if len(entities) > 0:
        reg = ':\[[E][0-9]\]'
        query = query.split()
        for i in range(len(query)):
            if re.match(reg, query[i]) is not None:
                query[i] = ':' + entities['[E0]'][0]
        query = ' '.join(query)
    return query

def caculate_f1(pred_ans, gold_ans):
    inter = set(pred_ans).intersection(set(gold_ans))
    recall = len(inter) / len(gold_ans)
    precision = 0 if len(pred_ans)==0 else len(inter) / len(pred_ans)
    if recall==0 and precision==0:
        f1 = 0
    else:
        f1 = 2* recall * precision / (recall + precision)
    return f1

def split_names(name):
    parts = name.split('.')
    out = []
    for part in parts:
        out.extend(part.split('_'))
    return out

def back_replace(word):
    if word.startswith(':.'):
        word = word.replace(':.', ':')
    elif word.startswith('rdf:.'):
        word = word.replace('rdf:.', 'rdf:')
    elif word.startswith('rdfs:.'):
        word = word.replace('rdfs:.', 'rdfs:')
    return word

def for_replace(word):
    if word.startswith(':'):
        word = word.replace(':', ':.')
    elif word.startswith('rdf:'):
        word = word.replace('rdf:', 'rdf:.')
    elif word.startswith('rdfs:'):
        word = word.replace('rdfs:', 'rdfs:.')
    return word

def get_n_hop(ent, n):
    out_ent = []
    out_rel = []
    sparql = SPARQLWrapper('http://localhost:23621/sparql')
    sparql.setReturnFormat(JSON)
    if n == 1:
        query = PREFIX + "SELECT ?p ?v WHERE { :%s ?p ?v . }" % ent
    elif n==2:
        query = PREFIX + "SELECT ?p ?v WHERE { :%s ?p1 ?v1 . ?v1 ?p ?v }" % ent
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        results = results['results']
    except:
        print("Bad query!!!")
        return [], []
    for r in results['bindings']:
        predicate1 = r['p']['value']
        entity1 = r['v']['value']
        if predicate1.startswith(PREFIX_DICT['rdf:']):
            predicate1 = predicate1.replace(PREFIX_DICT['rdf:'], 'rdf:')
        elif predicate1.startswith(PREFIX_DICT['rdfs:']):
            predicate1 = predicate1.replace(PREFIX_DICT['rdfs:'], 'rdfs:')
        elif predicate1.startswith(PREFIX_DICT[':']):
            predicate1 = predicate1.replace(PREFIX_DICT[':'], ':')
        if predicate1 not in out_rel and predicate1.startswith(':') and not predicate1.startswith(':m.') and not predicate1.startswith(':g.'):
            out_rel.append(predicate1)

        if entity1.startswith(PREFIX_DICT['rdf:']):
            entity1 = entity1.replace(PREFIX_DICT['rdf:'], 'rdf:')
        elif entity1.startswith(PREFIX_DICT['rdfs:']):
            entity1 = entity1.replace(PREFIX_DICT['rdfs:'], 'rdfs:')
        elif entity1.startswith(PREFIX_DICT[':']):
            entity1 = entity1.replace(PREFIX_DICT[':'], ':')
        if entity1 not in out_ent and entity1.startswith(':') and not entity1.startswith(':m.') and not entity1.startswith(':g.'):
            out_ent.append(entity1)
    #print("out rel number: %s" % len(out_rel))
    #print("out ent number: %s" % len(out_ent))
    #print(out_rel)
    #print(out_ent)
    #input()
    return out_ent, out_rel


def query_one_hop(src, sparql):
    out_ent = []
    out_rel = []
    #print("src ent number: %s" % len(src))
    for ent in src:
        query = PREFIX + "SELECT ?p1 ?v1 WHERE { %s ?p1 ?v1 }" % ent
        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
            results = results['results']
        except:
            continue
        for r in results['bindings']:
            predicate1 = r['p1']['value']
            entity1 = r['v1']['value']
            if predicate1.startswith(PREFIX_DICT['rdf:']):
                predicate1 = predicate1.replace(PREFIX_DICT['rdf:'], 'rdf:')
            elif predicate1.startswith(PREFIX_DICT['rdfs:']):
                predicate1 = predicate1.replace(PREFIX_DICT['rdfs:'], 'rdfs:')
            elif predicate1.startswith(PREFIX_DICT[':']):
                predicate1 = predicate1.replace(PREFIX_DICT[':'], ':')
            if predicate1 not in out_rel and predicate1.startswith(':'):
                out_rel.append(predicate1)

            if entity1.startswith(PREFIX_DICT['rdf:']):
                entity1 = entity1.replace(PREFIX_DICT['rdf:'], 'rdf:')
            elif entity1.startswith(PREFIX_DICT['rdfs:']):
                entity1 = entity1.replace(PREFIX_DICT['rdfs:'], 'rdfs:')
            elif entity1.startswith(PREFIX_DICT[':']):
                entity1 = entity1.replace(PREFIX_DICT[':'], ':')
            if entity1 not in out_ent and entity1.startswith(':'):
                out_ent.append(entity1)
    #print("out rel number: %s" % len(out_rel))
    #print("out ent number: %s" % len(out_ent))
    #print(out_ent)
    #input()
    return out_ent, out_rel


def relation_binding(pred, ind): #entities):
    '''
    sparql = SPARQLWrapper('http://localhost:23621/sparql')
    sparql.setReturnFormat(JSON)

    hops = 2
    #hops_rel = {}
    hops_ent = {}
    ent_all = []
    rel_all = []

    #hops_ent[0] = [ ':'+e[0] for k,e in entities.items()]
    hops_ent[0] = entities
    
    #ent_all.extend(hops_ent[0])
    for h in range(hops):
        next_hop_ent, next_hop_rel = query_one_hop(hops_ent[h], sparql)
        rel_all.extend(next_hop_rel)
        if len(set(next_hop_ent) - set(ent_all)) == 0: # can't find new ent
            break
        else:
            hops_ent[h+1] = []
            for ent in next_hop_ent:
                if ent not in ent_all:
                    hops_ent[h+1].append(ent)
                    #if not ent.startswith(':g.') and not ent.startswith(':m.'):
                    ent_all.append(ent)
    rel_all = list(set(rel_all))
    #ent_all = [ e for e in ent_all if not e.startswith(':g.') and not e.startswith(':m.')]
    
    #ent_all, rel_all = query_one_hop(hops_ent[0], sparql)
    rel_all = [ r for r in rel_all if (not r.startswith(':m.') and not r.startswith(':g.') ) ]
    ent_all = [ e for e in ent_all if (not e.startswith(':m.') and not e.startswith(':g.') ) ]
    #print(rel_all)
    #print(ent_all)
    #return ent_all, rel_all
    '''
    ent_all = ent2hop[ind][0]
    rel_all = ent2hop[ind][1]
    
    relations = [for_replace(r) for r in rel_all]
    entities = [for_replace(e) for e in ent_all]

    bm25_rel = BM25_Model([ split_names(r) for r in relations ])
    bm25_ent = BM25_Model([ split_names(e) for e in entities ])

    pred = pred.replace('(', ' ( ')
    pred = pred.replace(')', ' ) ')
    pred_sent = pred.split(' . ')
    triple_pos = -1
    out = []
    for sent in pred_sent:
        #print(sent)
        words = sent.split()
        for word in words:
            if word.isupper() or word in ['{', '}', 'rdfs:', 'rdf:', ':', '(', ')'] or word.startswith('<'):
                triple_pos = -1
                out.append(word)
            else:
                if word.startswith(':m.') or word.startswith(':g.'):
                    triple_pos = -1
                    out.append(word)
                elif word.startswith(':') or word.startswith('?') or word.startswith('rdf:') or word.startswith('rdfs:'):
                    triple_pos += 1

                    if word.startswith('?'):
                        out.append(word)
                        continue
                    elif word.startswith(':'):
                        word = word.replace(':', ':.')
                    elif word.startswith('rdf:'):
                        word = word.replace('rdf:', 'rdf:.')
                    elif word.startswith('rdfs:'):
                        word = word.replace('rdfs:', 'rdfs:.')

                    if triple_pos == 0 or triple_pos == 2:
                        #print('entity: %s' % word)
                        scores = bm25_ent.get_documents_score(split_names(word))
                        sort_index = np.argsort(scores)
                        replace = entities[sort_index[-1]]
                        #print('binded to %s' % replace)

                    elif triple_pos == 1:
                        #print('predicate: %s' % word)
                        scores = bm25_rel.get_documents_score(split_names(word))
                        sort_index = np.argsort(scores)
                        replace = relations[sort_index[-1]]
                        #print('binded to %s' % replace)
                    out.append(back_replace(replace))

                    if triple_pos >= 2:
                        triple_pos = -1
                else:
                    triple_pos = -1
                    out.append(word)
        #input()
    out = ' '.join(out)

    return out

if __name__=='__main__':
    # load preds and ans
    result_file = '/data/ljx/result/probeLLM/sparql/nl2lf_sparql_text-davinci-003_2023-06-28_demo15_121/pred.json'
    #result_file = '/home/ljx/new_cache_server32_0411/GrailQA_data/grailqa_v1.0_dev.json'
    #sampled_file = './sampled_grailqa_dev_120.json'
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    #with open(sampled_file, 'r') as f:
    #    sampled_index = json.load(f)

    #entlink_file = '/home/ljx/GrailQA-main/entity_linking/grailqa_el.json'
    #with open(entlink_file, 'r') as f:
    #    entlink = json.load(f)
    
    em_count = 0
    F1_score = []

    #data = data[:20]
    print("sampled data length: %s" % len(data))
    #grounding_file = './grailqa_dev_ent2hop_info.json'
    #groundings = []
    
    #for ind, index in enumerate(sampled_index):
    for ind, entry in enumerate(data):
        #entry = data[index]
        pred = entry['pred']
        #qid = str(entry['qid'])
        #entities = entlink[qid]['entities']
        #entities = [ ':' + e for e in entities.keys() ]
        entities = entry['entities']
        pred = preprocess(pred, entities)

        if len(entities)>0:
            pred = relation_binding(pred, ind) #entities)
        
        '''
        ent, rel = relation_binding(pred, entities)
        print("Example %s, index %s, sampled rel#num %s, sampled ent#num %s" % (ind, index, len(rel), len(ent)))
        groundings.append([ent, rel])
    with open(grounding_file, 'w') as f:
        json.dump(groundings, f)
        '''
        
        
        ans = [a['answer_argument'] for a in entry['answer']]
        #print(ans)
        gold_query = entry['sparql_query']
        gold_query = ' '.join(gold_query.split('\n'))
        if pred == gold_query:
            em_count += 1
        pred_ans = get_sparql_ans(pred)
        #print("query: ", pred)
        #print(pred_ans)
        #input()
        f = caculate_f1(pred_ans, ans)
        F1_score.append(f)
        #print("Index %s, f1: %s" % ( ind, f ))
    em_score = em_count / len(data)
    F1 = sum(F1_score) / len(data)
    f11 = sum(F1_score[:40])/40
    f12 = sum(F1_score[40:80])/40
    f13 = sum(F1_score[80:120])/40
    print("Exact match:", em_score)
    print("F1 score: ", F1)
    print("    sub groups: {:2f}, {:2f}, {:2f}".format(f11,f12,f13))
