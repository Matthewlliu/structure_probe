import json
import os

def process_seq(seq):
    seq = seq.replace('<pad>', '')
    seq = seq.split('</s>')[0]
    seq = seq.split('<unk>')[0]
    return seq.strip()

root = '/data/ljx/result/probeLLM/sparql/lf2nl_sparql_text-davinci-003_2023-06-20_44337'
original_src = '/home/ljx/new_cache_server32_0411/kqa_demonstration/structure_probe_codebase/cache/sparql/dataset'

file_num = 9
file_name = 'generated_ques_%s.json'

train_data = []
for i in range(file_num):
    file = os.path.join(root, file_name % i)
    with open(file, 'r') as f:
        tmp = json.load(f)
    train_data.extend(tmp)
print(len(train_data))

out_file = os.path.join(root, 'grailqa_v1.0_train.json')
#for entry in train_data:
#    entry['question'] = process_seq(entry['question'])

with open(out_file, 'w') as f:
    json.dump(train_data, f)

cmd = ' '.join(['cp' ,os.path.join(original_src, 'grailqa_v1.0_test_public.json'), root])
os.system(cmd)

cmd = ' '.join(['cp' ,os.path.join(original_src, 'grailqa_v1.0_dev.json'), root])
os.system(cmd)

#cmd = ' '.join(['cp' ,os.path.join(original_src, 'kb.json'), root])
#os.system(cmd)