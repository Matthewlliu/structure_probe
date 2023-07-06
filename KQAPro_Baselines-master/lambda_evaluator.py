import os
import json

def extract_bones(lf):
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

if __name__=='__main__':
    result_file = '/data/ljx/result/probeLLM/lambdaDCS/nl2lf_lambdaDCS_text-davinci-003_2023-06-28_demo30_240/pred.json'
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    '''
    sample_path = '/home/ljx/structure_probe/KQAPro_Baselines-master/sampled_overnight_test_240.json'
    with open(sample_path, 'r') as f:
        sample_id = json.load(f)

    aug_part = []
    for k, inds in sample_id.items():
        with open(os.path.join(args.test_dir, k+'_test.tsv'), 'r') as f:
            tmp = f.readlines()
            aug_part.extend([ tmp[i].strip() for i in inds ])
    gold_ans = []
    for a in aug_part:
        gold_ans.append(a.strip().split('\t'))[1]
    '''
    em_count = 0
    print("sampled data length: %s" % len(data))

    for entry in data:
        pred, ans = entry.strip().split('\t')
        pred = pred.strip()
        #pred = extract_bones(pred)
        #ans = extract_bones(ans)
        #print(pred)
        #print(ans)
        #input()
        if pred == ans:
            em_count += 1
    em_score = em_count / len(data)
    print("Exact match:", em_score)
