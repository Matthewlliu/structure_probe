from SPARQLWrapper import SPARQLWrapper, JSON
import os
import json
import re
from tqdm import tqdm

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
            if r['value']['type'] == 'uri':
                out.append(r['value']['value'].split('/')[-1])
            else:
                out.append(r['value']['value'])
    except:
        out = []
    return out

def preprocess(query, entities):
    prefix = 'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> '
    query = prefix + query 
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

if __name__=='__main__':
    # load preds and ans
    result_file = '/data/ljx/result/probeLLM/sparql/nl2lf_sparql_text-davinci-003_2023-06-12_2000/pred.json'
    #result_file = '/home/ljx/new_cache_server32_0411/GrailQA_data/grailqa_v1.0_dev.json'
    with open(result_file, 'r') as f:
        data = json.load(f)
    em_count = 0
    F1_score = []
    
    for entry in tqdm(data):
        pred = entry['pred']
        entities = entry['entities']
        pred = preprocess(pred, entities)
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
        if f>0:
            print(entry['question'])
            print(ans)
            print(pred_ans)
            input()
    em_score = em_count / len(data)
    F1_score = sum(F1_score) / len(data)
    print("Exact match:", em_score)
    print("F1 score: ", F1_score)
