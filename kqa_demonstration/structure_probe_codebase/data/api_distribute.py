import os

root = '/home/ljx/new_cache_server32_0411/kqa_demonstration/structure_probe_codebase/data'
source = os.path.join(root, 'api_keys_0418.txt')
with open(source, 'r') as f:
    allkeys = f.readlines()
    
allkeys = [ k.split('----')[-1].strip() for k in allkeys ]
assert len(allkeys)==20

step = 10
for i in range(2):
    s = i*step
    e = (i+1)*step
    keys = allkeys[s:e]
    name = 'api_keys_%s-%s.txt' % (step, i+1)
    with open(os.path.join(root, name), 'w') as f:
        for key in keys:
            f.write(key + '\n')
