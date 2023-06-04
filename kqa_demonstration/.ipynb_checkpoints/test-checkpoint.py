import os
import json
import numpy as np

test_src_file = '/data/ljx/data/KQAPro/dataset/val.json'
resource_file = 'cache/split_by_function/functions_ind.jsonl'
train_src_file = '/data/ljx/data/KQAPro/dataset/train.json'

with open(test_src_file, 'r') as f:
    test_question = json.load(f)
    
resource_data = {}
with open(resource_file, 'r') as f:
    for line in f:
        resource_data.update(json.loads(line))
        
with open(train_src_file, 'r') as f:
    train_data = json.load(f)

def ensemble_program_input(program):
    ret = ''
    for p in program:
        tmp = p['function']
        tmp += '(' + ', '.join(p['inputs']) + ')'
        ret += tmp
    return ret
    
def ensemble_input(demon, program):
    prompt = ''
    for pair in demon:
        prompt += pair[1] + ' is verbalized as: ' + pair[0] + '[SEP]'
    prompt += program + ' is verbalized as: '
    return prompt
    
for entry in test_question:
    print("Question:")
    print(entry['question'])
    program = entry['program']
    print(program)
    func_set = []
    for f in program:
        if f['function']=='Relate':
            func_set.append(f['function']+':'+f['inputs'][1])
        else:
            func_set.append(f['function'])
            
    func_seq = '<sep>'.join(func_set)
    if func_seq in resource_data:
        print(len(resource_data[func_seq]))
        ret_num = min(3, len(resource_data[func_seq]))
        samples = np.random.choice(resource_data[func_seq], ret_num, replace=False)
        pairs = []
        for ind in samples:
            que = train_data[ind]['question']
            pro = train_data[ind]['program']
            pro = ensemble_program_input(pro)
            #print(ind, ':', que)
            #print(' :', pro)
            pairs.append([que, pro])
        prompt = ensemble_input(pairs, ensemble_program_input(program))
        print(prompt)
    else:
        continue
    
    input()