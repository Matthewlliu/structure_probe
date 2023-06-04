from test_overnight import sending_post, pprint_res

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

from tqdm import tqdm

cached_train_path = 'cache/split_by_function/functions_ind.jsonl'
data = {}
with open(cached_train_path, 'r') as f:
    for line in f:
        data.update(json.loads(line))
keys = list(data.keys())

    
def extract_program_content(program):
    ret = []
    for p in program:
        ret.extend(p['inputs'])
    return ret

train_src_file = '/data/ljx/data/KQAPro/dataset/train.json'
with open(train_src_file, 'r') as f:
    train_data = json.load(f)
for entry in train_data:
    entry["all_input_content"] = extract_program_content(entry['program'])


def ensemble_program_input(program):
    ret = ''
    for p in program:
        tmp = p['function']
        tmp += '(' + ', '.join(p['inputs']) + ')'
        ret += tmp
    return ret
    
def post_process(res):
    return res[0][0].split('[SEP]')[0]
    
def ensemble_input(demon, program):
    prompt = ''
    for pair in demon:
        prompt += pair[1] + ' is verbalized as: ' + pair[0] + '[SEP]'
    prompt += program + ' is verbalized as: '
    return prompt
    
def program2funcset(program):
    func_set = []
    for f in program:
        if f['function']=='Relate':
            func_set.append(f['function']+':'+f['inputs'][1])
        else:
            func_set.append(f['function'])
    return func_set

# edit distance
def levenshtein(l1, l2, thresh=np.inf):
    """
        l1, l2: list of kopl functions
        thresh: maximum edit distance allowed minus 1
    """
    len1 = len(l1)
    len2 = len(l2)
    dp = np.zeros([len1+1, len2+1])
    for i in range(1, len1+1):
        dp[i][0] = i
    for j in range(1, len2+1):
        dp[0][j] = j
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if l1[i-1] == l2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = np.min([dp[i-1][j-1], dp[i-1][j], dp[i][j-1]]) + 1
            #if dp[i][j] > thresh + 1:
            #    return dp[i][j]
    return int(dp[len1][len2])

def find_max_cover(str1, dis_dict, sup=None):
    """
        dis_dict: List of keys(bones) ids of the closest distance, usually 0 or 1
    """
    s1 = set(str1)
    cand = []
    diff = {}
    
    for key_ind in dis_dict:
        s2 = set(keys[key_ind].split('<sep>'))
        d = s1 - s2
        diff[key_ind] = len(d)
        #if len(d) == 0:
        #    return [key_ind]
    # sort by the difference
    diff = sorted(diff.items(), key = lambda kv:(kv[1], kv[0]))
    # if there are 0-dist program (complete covered)
    ccand = [ d[0] for d in diff if d[1]==0 ]
    
    if len(ccand)==1: # only one completely coveres s1, in case no enough example
        if len(data[keys[ccand[0]]]) >= 3:
            return ccand
    elif len(ccand) > 1:
        num = np.min([3, len(ccand)])
        return np.random.choice(ccand, num, replace=False).tolist()
    if_cover = False
    while(if_cover is False):
        max_subset = diff[0][0] # key index
        cand.append(max_subset)
        dis_dict = [d for d in dis_dict if d!=max_subset] # update dis_dict
        s1 = s1 - set(keys[max_subset].split('<sep>')) # uncovered function set
        
        if len(cand)>=3:
            break
            
        if len(s1)<1:
            if_cover = True
        else:
            diff = {}
            if_in = False
            for key_ind in dis_dict:
                s2 = set(keys[key_ind].split('<sep>'))
                if len(s1&s2) > 0:
                    if_in = True
                d = s1 - s2
                diff[key_ind] = len(d)
            diff = sorted(diff.items(), key = lambda kv:(kv[1], kv[0]))
            
            if if_in is False:
                if_cover=True # no candidate can cover the rest, break
    
    if sup is not None and len(cand)<3 and len(s1)>0:
        diff = {}
        if_in = False
        for key_ind in sup:
            s2 = set(keys[key_ind].split('<sep>'))
            if len(s1&s2) > 0:
                if_in=True
            d = s1 - s2
            diff[key_ind] = len(d)
        diff = sorted(diff.items(), key = lambda kv:(kv[1], kv[0]))
        if if_in is True:
            cand.append(diff[0][0])
    return cand

def find_content_related_example(sample_list, content, num, self_id):
    """
        sample_list: list of train_data inds to choose from
        content: set of input args
        num: sample num
    """
    recall = {}
    for sample_ind in sample_list:
        t_cont = train_data[sample_ind]["all_input_content"]
        recall[sample_ind] = len(content.intersection(set(t_cont)))/len(content)
    recall = sorted(recall.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    if self_id in sample_list:
        recall = recall[1:]
    ret = recall[:5]
    #print(ret)
    ret = [ r[0] for r in ret ] # only need the id
    return np.random.choice(ret, min(num, len(ret)), replace=False).tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment_size', type=int,
                        default=20)
    parser.add_argument('--save_step', type=int,
                        default=5000)
    args = parser.parse_args()
    
    
    #aug_data = train_data[:args.augment_size]
    print("Train data size: ", len(train_data))
    
    # start from 55k, and the former results are store in aug_train_10.json
    # new storing file is aug_train_rest_%s.json
    
    start_id = 5000 * 18
    for sample_id in tqdm(range(start_id, len(train_data))):
        entry = train_data[sample_id]
        program = entry['program']
        content = extract_program_content(program)
        func_set = program2funcset(program)
        
        dis_dict = {}
        for ind, funcs in enumerate(keys):
            funcs = funcs.split('<sep>')
            dist = levenshtein(func_set, funcs)
            if dist not in dis_dict:
                dis_dict[dist] = [ind]
            else:
                dis_dict[dist].append(ind)
        
        sorted_dist = sorted(dis_dict.keys())
        if sorted_dist[0]==0 and len(data[keys[dis_dict[sorted_dist[0]][0]]])<2: #完全相同的example数量不足
            arg1 = dis_dict[sorted_dist[1]]
            arg2 = dis_dict[sorted_dist[2]]
        else:
            arg1 = dis_dict[sorted_dist[0]]
            arg2 = dis_dict[sorted_dist[1]]
        cand = find_max_cover(func_set, arg1, sup=arg2)
        
        sample_num = []
        for i in range(len(cand)):
            if i == 0:
                sample_num.append(4 - len(cand))
            else:
                sample_num.append(1)
        
        pairs = []
        sample_cand = []
        more = 0
        for ind, ca in enumerate(cand):
            sample_list = data[keys[ca]]
            num = sample_num[ind] + more
            more = 0
            if len(sample_list)<num:
                more = num - len(sample_list)
                num = len(sample_list)
            #samples = np.random.choice(sample_list, num, replace=False).tolist()
            samples = find_content_related_example(sample_list, set(content), num, sample_id)
            sample_cand.extend(samples)
            for sample in samples:
                que = train_data[sample]['question']
                pro = train_data[sample]['program']
                pro = ensemble_program_input(pro)
                pairs.append([que, pro])
        prompt = ensemble_input(pairs, ensemble_program_input(program))
        #print(prompt)
        #input()
        res, time = sending_post([prompt], strategy = "BeamSearchStrategy")
        #pprint_res(res, [prompt], time)
        #print("Ref: %s" % entry["question"])
        #input()
        res = post_process(res)
        #print(res)
        if "aug_questions" in entry:
            entry["aug_questions"].append(res)
        else:
            entry["aug_questions"] = [res]
        
        if (sample_id+1) % args.save_step == 0 or (sample_id+1)==len(train_data):
            save_id = sample_id//args.save_step
            with open("/data/ljx/data/KQAPro/dataset/aug_train_rest_%s.json" % save_id, 'w') as f:
                json.dump(train_data[save_id*5000 : (save_id+1)*5000], f)

if __name__=='__main__':
    main()