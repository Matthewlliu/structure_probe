import os

root = '/Users/matthewliu/code/structure_probe_codebase/data'
source = os.path.join(root, 'api_keys_0327.txt')
with open(source, 'r') as f:
    allkeys = f.readlines()
    
allkeys = [ k.split('----')[-1].strip() for k in allkeys ]
assert len(allkeys)==64

step = 16
for i in range(4):
    s = i*step
    e = (i+1)*step
    keys = allkeys[s:e]
    name = 'api_keys_16-%s.txt' % (i+1)
    with open(os.path.join(root, name), 'w') as f:
        for key in keys:
            f.write(key + '\n')
