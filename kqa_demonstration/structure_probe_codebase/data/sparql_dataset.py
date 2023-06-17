import os
import json
from utils import levenshtein, ensemble_input
import numpy as np
from bm25 import BM25_Model
from tqdm import tqdm

s_symbols = ["(", ")", "AND", "JOIN", "ARGMIN", "ARGMAX", "R", "COUNT"]
#mid2ent_file = '/home/ljx/new_cache_server32_0411/GrailQA_data/mid2name.tsv'
mid2ent_file = '/home/ljx/entity_list_file_freebase_complete_all_mention'
entlink_file = '/home/ljx/GrailQA-main/entity_linking/grailqa_el.json'
doc_file = '/home/ljx/new_cache_server32_0411/GrailQA_data/doc_file.json'

class sparql_data(object):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_dir
        self.cache_path = args.cache_dir
        
        # initialize, load training data
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
            
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
            
            # build content
            self.content = []
            for entry in self.data:
                self.content.append(self.extract_content(entry['s_expression']))

            # build mid2name
            self.mid2name = {}
            with open(mid2ent_file, 'r') as f:
                for line in f.readlines():
                    tmp = line.split('\t')
                    mid = tmp[0]
                    name = tmp[1]
                    self.mid2name[mid] = name
        else:
            with open(entlink_file, 'r') as f:
                self.grailqa_el = json.load(f)

            if os.path.exists(doc_file):
                self.docs_list = json.load(open(doc_file, 'r'))
            else:
                self.mid2name = {}
                with open(mid2ent_file, 'r') as f:
                    for line in f.readlines():
                        tmp = line.split('\t')
                        mid = tmp[0]
                        name = tmp[1]
                        self.mid2name[mid] = name
                self.docs_list = []
                for d in tqdm(self.data):
                    q, sp = self.doc_anonymize(d)
                    self.docs_list.append([q, sp])
                with open(doc_file, 'w') as f:
                    json.dump(self.docs_list, f)
                print('Documents cache saved')
            
            docs = [ d[0] for d in self.docs_list]
            self.bm25_model = BM25_Model(docs)
        #print(self.mid2name['m.0j6v9fs'])
        #print(self.mid2name['m.020mfr'])
        #print(self.mid2name['m.025sx5b'])
        #exit()

    def doc_anonymize(self, d):
        question = d['question']
        sp = d['sparql_query'].split('\n')[1:]
        sp = ' '.join(sp)
        sp = sp.split()

        count = 0
        template = '[E%s]'
        for i in range(len(sp)):
            part = sp[i]
            if part.startswith(":m."):
                mid = part[1:]
                name = self.mid2name[mid]
                question = question.replace(name, template % count)
                sp[i] = ':'+template % count
                count += 1
        return question.split(), ' '.join(sp)
    
    def test_anonymize(self, d):
        """
            return:
                --question
        """
        #sp = d['sparql_query']
        qid = str(d['qid'])
        question = self.grailqa_el[qid]['question']

        #sp = sp.split('\n')[1:]
        #sp = ' '.join(sp)
        template = '[E%s]'
        #question += " (the entity in the question are: "
        entities = {}
        for ind, ent_id in enumerate(self.grailqa_el[qid]['entities'].keys()):
            #sp = sp.replace( ent_id, template % ind )
            question = question.replace(self.grailqa_el[qid]['entities'][ent_id]['mention'], template % ind)
            #question += template % ind + ' is %s, ' % self.grailqa_el[qid]['entities'][ent_id]['mention']
            entities[template%ind] = [ent_id, self.grailqa_el[qid]['entities'][ent_id]['mention']]
        #question += ' ) '
        return question, entities
            
    def build_cache_from_scratch(self):
        cache_path = '/'.join(self.cache_path.split('/')[:-1])
        os.makedirs(cache_path, exist_ok = True)
        
        cache = {}
        for ind, entry in enumerate(self.data):
            #sparql = entry['sparql']
            s_expression = entry['s_expression']
            
            function_seq = self.extract_bones(s_expression)
            if function_seq in cache:
                cache[function_seq].append(ind)
            else:
                cache[function_seq] = [ind]
        return cache
    
    def extract_bones(self, lf):
        s = lf.replace('(', ' ( ').replace(')', ' ) ')
        s = s.split()

        out = []
        for ss in s:
            if ss in s_symbols:
                out.append(ss)
        return ' '.join(out)

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

    def save_cache(self):
        with open(self.cache_path, 'w') as f:
            for k,v in self.cache.items():
                json.dump({k:v}, f)
                f.write('\n')
                
    def extract_content(self, lf):
        s = lf.replace('(', ' ( ').replace(')', ' ) ')
        s = s.split()

        ret = []
        for ss in s:
            if ss not in s_symbols:
                if ss.startswith('m.'):
                    continue
                tmp = ss.split('.')
                for i in range(len(tmp)):
                    ret.append('.'.join(tmp[:i+1]))
        return ret
    
    def retrieve_demonstrations(self, entry, sample_id):
        """
            sample_id: 用于排除自身
        """
        if self.args.if_lf2nl:
            sparql = entry['sparql_query']
            s_expression = entry['s_expression']
            content = self.extract_content(s_expression)
            skeleton = self.extract_bones(s_expression)

            
            dis_dict = {}
            for ind, funcs in enumerate(self.keys):
                funcs = funcs.split(' ')
                dist = levenshtein(skeleton.split(), funcs)
                if dist not in dis_dict:
                    dis_dict[dist] = [ind]
                else:
                    dis_dict[dist].append(ind)
                    
            sorted_dist = sorted(dis_dict.keys())

            # select the closest keys(skeletons)' ids
            cand = []
            if sorted_dist[0]==0 and len(self.cache[self.keys[dis_dict[sorted_dist[0]][0]]])<3:
                # not likely, usually there are more than 3 examples for each skeleton
                cand.append(dis_dict[sorted_dist[0]][0])
                cand.extend(np.random.choice(dis_dict[sorted_dist[1]], 1).tolist())
            else:
                # sorted_dist[0]==0 actually is always true
                cand.append(dis_dict[sorted_dist[0]][0])
            
            pairs = self.sample_candidates(cand, content, sample_id, self.args.demo_num)
            prompt = ensemble_input(pairs, self.sparql_preprocess(sparql), self.args.logic_forms, self.args.if_lf2nl, reverse=True)
            return prompt, None
        else:
            question = entry['question']
            scores = self.bm25_model.get_documents_score(question)
            sort_index = np.argsort(scores)
            cand = sort_index[-self.args.demo_num:]
            pairs = []
            for sample in cand:
                que, sp = self.docs_list[sample]
                que = ' '.join(que)
                pairs.append([que, sp])
            que, entities = self.test_anonymize(entry)
            prompt = ensemble_input(pairs, que, self.args.logic_forms, self.args.if_lf2nl, reverse=True)
            return prompt, entities

    def sparql_preprocess(self, sp):
        sp = sp.split('\n')[1:]
        sp = ' '.join(sp)
        sp = sp.split()
        for i in range(len(sp)):
            part = sp[i]
            if part.startswith(":m."):
                mid = part[1:]
                name = self.mid2name[mid]
                sp[i] = ':'+name
        return ' '.join(sp)

    def sample_candidates(self, cand, content, sample_id, demo_num):
        """
            cand: List of candidates of skeleton index
        """
        # sample
        sample_num = []
        assert len(cand) == 1 or len(cand) == 2
        if len(cand) == 1:
            sample_num = [demo_num]
        else:
            sample_num = [len(self.cache[self.keys[cand[0]]])-1, 1]
                
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
            pro = self.data[sample]['sparql_query']
            pro = self.sparql_preprocess(pro)
            pairs.append([que, pro])
        return pairs