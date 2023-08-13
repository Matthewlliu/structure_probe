import json
import os
import numpy as np
'''
from SPARQLWrapper import SPARQLWrapper, JSON

PREFIX_DICT = {
    "rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
    ":": "http://rdf.freebase.com/ns/"
}
PREFIX = 'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> '

path = '/home/ljx/new_cache_server32_0411/GrailQA_data/grailqa_v1.0_dev.json'
with open(path, 'r') as f:
    data = json.load(f)

path = '/home/ljx/structure_probe/KQAPro_Baselines-master/sampled_grailqa_dev_120.json'
with open(path, 'r') as f:
    index = json.load(f)

entlink_file = '/home/ljx/GrailQA-main/entity_linking/grailqa_el.json'
with open(entlink_file, 'r') as f:
    entlink = json.load(f)

sparql = SPARQLWrapper('http://localhost:23621/sparql')
sparql.setReturnFormat(JSON)

i = 0
while i<len(index):
    entry = data[index[i]]
    qid = str(entry['qid'])
    entities = entlink[qid]['entities']
    if len(entities)>1 or len(entities) == 0:
        index[i] += 1
        print("Example %s, index %s, skipped" % (i, index[i]))
        continue
    ent = ':' + list(entities.keys())[0]
    query = PREFIX + "SELECT ?p1 ?v1 WHERE { %s ?p1 ?v1 }" % ent
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        results = results['results']
    except:
        continue
    out_ent = []
    out_rel = []
    for r in results['bindings']:
        predicate1 = r['p1']['value']
        entity1 = r['v1']['value']

        if entity1.startswith(PREFIX_DICT['rdf:']):
            entity1 = entity1.replace(PREFIX_DICT['rdf:'], 'rdf:')
        elif entity1.startswith(PREFIX_DICT['rdfs:']):
            entity1 = entity1.replace(PREFIX_DICT['rdfs:'], 'rdfs:')
        elif entity1.startswith(PREFIX_DICT[':']):
            entity1 = entity1.replace(PREFIX_DICT[':'], ':')
        if entity1 not in out_ent and entity1.startswith(':'):
            out_ent.append(entity1)
    print("Example %s, index %s, one-hop entity number: %s" % (i, index[i], len(out_ent)))
    if len(out_ent) > 30:
        index[i] += 1
        continue
    i += 1
with open('/home/ljx/structure_probe/KQAPro_Baselines-master/sampled_grailqa_dev_120.json', 'w') as f:
    json.dump(index, f)
    '''

'''
domains = ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']
example_num = [391, 399, 168, 189, 161, 216, 332, 884]
sample_num = 30

path = '/home/ljx/semantic-parsing-dual-master/data/overnight'
out = {}
for i,domain in enumerate(domains):
    #file = os.path.join(path, domain+'_test.tsv')
    x = list(range(example_num[i]))
    samples = np.random.choice(x, sample_num, replace=False).tolist()
    samples = sorted(samples)
    out[domain] = samples
out_file = './sampled_overnight_test_240.json'
with open(out_file, 'w') as f:
    json.dump(out, f)
'''

path = '/home/ljx/new_cache_server32_0411/KQAPro/dataset/val.json'
with open(path, 'r') as f:
    data = json.load(f)
x = list(range(len(data)))
samples = np.random.choice(x, 300, replace=False).tolist()
samples = sorted(samples)
out_file = './sampled_kopl_test_300.json'
with open(out_file, 'w') as f:
    json.dump(samples, f)