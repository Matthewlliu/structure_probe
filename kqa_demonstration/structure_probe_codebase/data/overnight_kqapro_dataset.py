import os
import json
from utils import levenshtein, ensemble_input
import numpy as np
import re
from bm25 import BM25_Model

domains = ['calendar', 'blocks', 'housing', 'restaurants', 'publications', 'recipes', 'social', 'basketball']
lex_file = '/home/ljx/semantic-parsing-dual-master/data/overnight/human/all_lexicon.txt'
lexicon = []
with open(lex_file, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        parts = line.split(':')
        lexicon.append([parts[0], parts[2]])

class overnight_kqapro_data(object):
    def __init__(self, args):
        self.args = args
        self.seed_path = args.seed_dir
        self.aug_path = args.data_dir
        self.cache_path = args.cache_dir

        # initialize, load training data
        '''if os.path.isdir(self.data_path):
            train_files = [ d + '.paraphrases.train.examples' for d in domain ]
            all_data_file = os.path.join(self.data_path, 'all_train_data.json')'''
            
        with open(self.seed_path, 'r') as f:
            self.data = json.load(f)

        with open(self.aug_path, 'r') as f:
            self.aug_part = json.load(f)

        if self.args.if_lf2nl:
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

            # extract content
            self.content = []
            for entry in self.data:
                lf = entry['lambda-dcs']
                #_, lf = entry.strip().split('\t')
                self.content.append(self.extract_content(lf))
        else:
            self.docs_list = [self.entity_anonymize(d) for d in self.data]
            self.bm25_model = BM25_Model(self.docs_list)

    def entity_anonymize(self, entry):
        question, _ = entry.strip().split('\t')
        for e in lexicon:
            if e[0] in question:
                question = question.replace(e[0], '[E]')
        return question.split()

    def build_cache_from_scratch(self):
        cache_path = '/'.join(self.cache_path.split('/')[:-1])
        os.makedirs(cache_path, exist_ok = True)
        
        cache = {}
        for ind, entry in enumerate(self.data):
            lf = entry['lambda-dcs']
            bones = self.extract_bones(lf)
            if bones in cache:
                cache[bones].append(ind)
            else:
                cache[bones] = [ind]
        return cache

    def save_cache(self):
        with open(self.cache_path, 'w') as f:
            for k,v in self.cache.items():
                json.dump({k:v}, f)
                f.write('\n')

    def extract_bones(self, lf):
        bone_names = ['call', 'string', 'lambda', 'var', 'date', 'number', 'time', '+']
        lf = lf.split()
        out = []
        
        for term in lf:
            if term in '(':
                out.append(term)
            elif term in ')':
                if out[-1] in '(':
                    out = out[:-1]
                else:
                    out.append(term)
            elif term in bone_names or term.startswith('SW.') or term.startswith('.'):
                out.append(term)
        return ' '.join(out)

    def extract_content(self, lf):
        bone_names = ['call', 'string', 'lambda', 'var', 'date', 'number', 'time', '+', '(', ')']
        lf = lf.split()
        out = []

        for term in lf:
            if term in bone_names or term.startswith('SW.') or term.startswith('.'):
                continue
            else:
                out.append(term)
        return out

    def retrieve_demonstrations(self, entry, sample_id):
        """
            sample_id: 用于排除自身
        """
        if self.args.if_lf2nl:
            text = entry['rewrite']
            lf = entry['lambda-dcs']

            bones = self.extract_bones(lf)
            content = self.extract_content(lf)

            closest_bones = {}
            min_dist = 1e5
            for func in self.cache.keys():
                dist = levenshtein(func.split(), bones.split(), min_dist)
                if dist < min_dist:
                    min_dist = dist
                    closest_bones[dist] = [func]
                elif dist == min_dist:
                    closest_bones[dist].append(func)

            #examples = []
            example_func = np.random.choice(closest_bones[min_dist], 1, replace=False)

            if len(self.cache[example_func[0]]) > self.args.demo_num: # 不是等于，很有可能包含自己
                #examples = np.random.choice(self.cache[example_func[0]], 2, replace=False
                #print("first")
                pass
            else:
                #print("second")
                #examples = self.cache[example_func[0]]
                if len(closest_bones[min_dist])>1: # 这种情况不太能发生
                    #print("1")
                    example_func = np.random.choice(closest_bones[min_dist], 2, replace=False)
                else:
                    #print("2")
                    dists = sorted(list(closest_bones.keys()))
                    another_func = np.random.choice(closest_bones[dists[1]], 1)
                    example_func.extend(another_func)
            #print(bones)
            #print(example_func)
            pairs = self.sample_candidates(example_func, content, sample_id, self.args.demo_num)

            prompt = ensemble_input(pairs, lf, self.args.logic_forms, self.args.if_lf2nl, reverse=True)
            return prompt, None
        else:
            question, lf = entry.strip().split('\t')
            scores = self.bm25_model.get_documents_score(question)
            sort_index = np.argsort(scores)
            cand = sort_index[-self.args.demo_num:]
            pairs = []
            for sample in cand:
                que, lf = self.data[sample].strip().split('\t')
                #lf = self.extract_bones(lf)
                pairs.append([que, lf])
            prompt = ensemble_input(pairs, question, self.args.logic_forms, self.args.if_lf2nl, reverse=True)
            
            #entity = []
            #for e in lexicon:
            #    if e[0] in question:
            #        entity.append(e)
            return prompt, None #, entity


    def sample_candidates(self, cand, content, sample_id, demo_num=2):
        """
            cand: List of candidates of skeleton index
        """
        sample_num = []
        for i in range(len(cand)):
            if i == 0:
                sample_num.append(demo_num+1 - len(cand))
            else:
                sample_num.append(1)
        #print(cand)
        #print(sample_num)
        pairs = []
        sample_cand = []
        more = 0
        for ind, ca in enumerate(cand):
            sample_list = self.cache[ca]
            num = sample_num[ind] + more
            more = 0
            if len(sample_list)<num:
                more = num - len(sample_list)
                num = len(sample_list)
            samples = self.find_content_related_example(sample_list, set(content), num, sample_id)
            sample_cand.extend(samples)
        if more > 0:
            print("Warning, not enough examples. Require {}, retrieved {}, {} short".format(self.args.demo_num, len(sample_cand), more))
        for sample in sample_cand:
            #que, pro = self.data[sample].split('\t')
            que = self.data[sample]['rewrite']
            pro = self.data[sample]['lambda-dcs']
            #pro = self.data[sample]['program']
            pairs.append([que, pro])
        return pairs

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
            t_cont = self.content[sample_ind]
            recall[sample_ind] = len(content.intersection(set(t_cont)))/len(content)
        recall = sorted(recall.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        
        #ret = recall[:5] #TODO
        recall = [ r[0] for r in recall ] # only need the id
        return recall[:min(num, len(recall))]

    def lf_pretifier(self, lf):
        lf = lf.split()
        count = -1
        out = []
        con = True
        for item in lf:
            if item == '(':
                count += 1
                out.append('\t'*count + '{')
                con = False
            elif item == ')':
                out.append('\t'*count + '}')
                count -= 1
                con = False
            else:
                if con:
                    out[-1] += ' ' + item
                else:
                    out.append('\t'*count + item)
                con = True
        return out

    def pprint(self, List):
        for l in List:
            print(l)