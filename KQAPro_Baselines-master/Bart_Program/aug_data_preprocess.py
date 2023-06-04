import json
import os

root = '/data/ljx/result/probeLLM/kopl/lf2nl_kopl_text-davinci-003_2023-03-27_94376'
original_src = '/data/ljx/data/KQAPro/dataset'

file_num = 19
file_name = 'generated_ques_%s.json'

train_data = []
for i in range(file_num):
    file = os.path.join(root, file_name % i)
    with open(file, 'r') as f:
        tmp = json.load(f)
    train_data.extend(tmp)
print(len(train_data))

out_file = os.path.join(root, 'train.json')
with open(out_file, 'w') as f:
    json.dump(train_data, f)

cmd = ' '.join(['cp' ,os.path.join(original_src, 'test.json'), root])
os.system(cmd)

cmd = ' '.join(['cp' ,os.path.join(original_src, 'val.json'), root])
os.system(cmd)

cmd = ' '.join(['cp' ,os.path.join(original_src, 'kb.json'), root])
os.system(cmd)