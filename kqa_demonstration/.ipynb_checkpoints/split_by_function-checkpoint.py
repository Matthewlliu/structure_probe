import os
import json

data_src_file = ['/data/ljx/data/KQAPro/dataset/train.json', ] #'/data/ljx/data/KQAPro/dataset/val.json']
cache_path = 'cache/split_by_function'
file_name = 'functions_ind.jsonl'

os.makedirs(cache_path, exist_ok = True)

data = []
for data_file in data_src_file:
    with open(data_file, 'r') as f:
        data.extend(json.load(f))
    
out = {}

for ind, entry in enumerate(data):
    program = entry['program']
    #function_set = [ f['function'] for f in program ]
    function_set = []
    for f in program:
        if f['function']=='Relate':
            function_set.append(f['function']+':'+f['inputs'][1])
        else:
            function_set.append(f['function'])
    function_seq = '<sep>'.join(function_set)
    if function_seq in out:
        out[function_seq].append(ind)
    else:
        out[function_seq] = [ind]

with open(os.path.join(cache_path, file_name), 'w') as f:
    for k,v in out.items():
        json.dump({k:v}, f)
        f.write('\n')