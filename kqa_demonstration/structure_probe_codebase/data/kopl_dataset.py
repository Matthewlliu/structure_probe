import os
import json
from utils import levenshtein, ensemble_input
import numpy as np

class kopl_data(object):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_dir
        self.cache_path = args.cache_dir
        
        # initialize, load training data
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
            
        # extract skeleton
        if not os.path.exists(self.cache_path):
            self.cache = self.build_cache_from_scratch()
            self.save_cache()
        else:
            self.cache = {}
            with open(self.cache_path, 'r') as f:
                for line in f:
                    self.cache.update(json.loads(line))
        self.keys = list(self.cache.keys())
        
        # build content
        for entry in self.data:
            entry["all_input_content"] = self.extract_program_content(entry['program'])

    def __len__(self):
        return len(self.data)
            
    def build_cache_from_scratch(self):
        cache_path = '/'.join(self.cache_path.split('/')[:-1])
        os.makedirs(cache_path, exist_ok = True)
        
        cache = {}
        for ind, entry in enumerate(self.data):
            program = entry['program']
            function_set = []
            for f in program:
                if f['function']=='Relate':
                    function_set.append(f['function']+':'+f['inputs'][1])
                else:
                    function_set.append(f['function'])
            function_seq = '<sep>'.join(function_set)
            if function_seq in cache:
                cache[function_seq].append(ind)
            else:
                cache[function_seq] = [ind]
        return cache
    
    def find_content_related_example(self, sample_list, content, num, self_id):
        """
            sample_list: list of train_data inds to choose from
            content: set of input args
            num: sample num
        """
        recall = {}
        if self_id in sample_list:
            sample_list = [s for s in sample_list if s != self_id]
        for sample_ind in sample_list:
            t_cont = self.data[sample_ind]["all_input_content"]
            recall[sample_ind] = len(content.intersection(set(t_cont)))/len(content)
        recall = sorted(recall.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

        #ret = recall[:5] #TODO
        recall = [ r[0] for r in recall ] # only need the id
        return recall[:min(num, len(recall))]

    def find_max_cover(self, str1, dis_dict, sup=None, demo_num=3):
        """
            dis_dict: List of keys(bones) ids of the closest distance, usually 0 or 1
        """
        s1 = set(str1)
        cand = []
        diff = {}

        for key_ind in dis_dict:
            s2 = set(self.keys[key_ind].split('<sep>'))
            d = s1 - s2
            diff[key_ind] = len(d)
            #if len(d) == 0:
            #    return [key_ind]
        # sort by the difference
        diff = sorted(diff.items(), key = lambda kv:(kv[1], kv[0]))
        # if there are 0-dist program (complete covered)
        ccand = [ d[0] for d in diff if d[1]==0 ]

        if len(ccand)==1: # only one completely coveres s1, in case no enough example
            if len(self.cache[self.keys[ccand[0]]]) >= demo_num:
                return ccand
        elif len(ccand) > 1:
            num = np.min([demo_num, len(ccand)])
            return np.random.choice(ccand, num, replace=False).tolist()
        if_cover = False
        while(if_cover is False):
            max_subset = diff[0][0] # key index
            cand.append(max_subset)
            dis_dict = [d for d in dis_dict if d!=max_subset] # update dis_dict
            s1 = s1 - set(self.keys[max_subset].split('<sep>')) # uncovered function set

            if len(cand)>=demo_num:
                break

            if len(s1)<1:
                if_cover = True
            else:
                diff = {}
                if_in = False
                for key_ind in dis_dict:
                    s2 = set(self.keys[key_ind].split('<sep>'))
                    if len(s1&s2) > 0:
                        if_in = True
                    d = s1 - s2
                    diff[key_ind] = len(d)
                diff = sorted(diff.items(), key = lambda kv:(kv[1], kv[0]))

                if if_in is False:
                    if_cover=True # no candidate can cover the rest, break

        if sup is not None and len(cand)<demo_num and len(s1)>0:
            diff = {}
            if_in = False
            for key_ind in sup:
                s2 = set(self.keys[key_ind].split('<sep>'))
                if len(s1&s2) > 0:
                    if_in=True
                d = s1 - s2
                diff[key_ind] = len(d)
            diff = sorted(diff.items(), key = lambda kv:(kv[1], kv[0]))
            if if_in is True:
                cand.append(diff[0][0])
        return cand
    
    def save_cache(self):
        with open(self.cache_path, 'w') as f:
            for k,v in self.cache.items():
                json.dump({k:v}, f)
                f.write('\n')
                
    def extract_program_content(self, program):
        ret = []
        for p in program:
            ret.extend(p['inputs'])
        return ret
    
    def program_serialize(self, program):
        ret = ''
        for p in program:
            tmp = p['function']
            tmp += '(' + ', '.join(p['inputs']) + ')'
            ret += tmp
        return ret
    
    def program2funcset(self, program):
        func_set = []
        for f in program:
            if f['function']=='Relate':
                func_set.append(f['function']+':'+f['inputs'][1])
            else:
                func_set.append(f['function'])
        return func_set
    
    def retrieve_demonstrations(self, entry, sample_id):
        """
            sample_id: 用于排除自身
        """
        program = entry['program']
        content = self.extract_program_content(program)
        func_set = self.program2funcset(program)
        
        dis_dict = {}
        for ind, funcs in enumerate(self.keys):
            funcs = funcs.split('<sep>')
            dist = levenshtein(func_set, funcs)
            if dist not in dis_dict:
                dis_dict[dist] = [ind]
            else:
                dis_dict[dist].append(ind)
                
        sorted_dist = sorted(dis_dict.keys())
        if sorted_dist[0]==0 and len(self.cache[self.keys[dis_dict[sorted_dist[0]][0]]])<2: #完全相同的example数量不足
            arg1 = dis_dict[sorted_dist[1]]
            arg2 = dis_dict[sorted_dist[2]]
        else:
            arg1 = dis_dict[sorted_dist[0]]
            arg2 = dis_dict[sorted_dist[1]]
        cand = self.find_max_cover(func_set, arg1, sup=arg2, demo_num=self.args.demo_num)
        
        pairs = self.sample_candidates(cand, content, sample_id, self.args.demo_num)
        prompt = ensemble_input(pairs, self.program_serialize(program), self.args.logic_forms, reverse=True)
        return prompt
        
    def sample_candidates(self, cand, content, sample_id, demo_num):
        """
            cand: List of candidates of skeleton index
        """
        # sample
        sample_num = []
        for i in range(len(cand)):
            if i == 0:
                sample_num.append(demo_num+1 - len(cand))
            else:
                sample_num.append(1)
                
        pairs = []
        sample_cand = []
        more = 0
        for ind, ca in enumerate(cand):
            sample_list = self.cache[self.keys[ca]]
            num = sample_num[ind] + more
            more = 0
            if len(sample_list)<num:
                more = num - len(sample_list)
                num = len(sample_list)
            #samples = np.random.choice(sample_list, num, replace=False).tolist()
            samples = self.find_content_related_example(sample_list, set(content), num, sample_id)
            sample_cand.extend(samples)
        if more > 0:
            print("Warning, not enough examples. Require {}, retrieved {}, {} short".format(self.args.demo_num, len(sample_cand), more))
        for sample in sample_cand:
            que = self.data[sample]['question']
            pro = self.data[sample]['program']
            pro = self.program_serialize(pro)
            pairs.append([que, pro])
        return pairs