import json
import copy
from tqdm import tqdm

data_file = '/data/ljx/data/KQAPro/KQAPro.json'
#data_file = '/data/ljx/data/KQAPro/dataset/train.json'

with open(data_file, 'r') as f:
    data = json.load(f)

FIND = ['Find']
RELATE = ['Relate']
FILTER = ['FilterConcept', 'FilterStr', 'FilterNum', 'FilterYear', 'FilterDate']
QUERY = ['QueryAttr', 'Count', 'What']

out = []
for entry in tqdm(data):
    program = entry['program']
    if len(program) == 3 or len(program) == 4:
        for ind, func in enumerate(program):
            if ind == 0:
                if func["function"] not in FIND:
                    break
            elif ind == 1:
                if func["function"] not in RELATE:
                    break
            elif ind == 2:
                if func["function"] in QUERY and len(program) == 3:
                    out.append(copy.deepcopy(entry))
                elif func["function"] not in FILTER:
                    break
            elif ind == 3:
                if func["function"] in QUERY:
                    out.append(copy.deepcopy(entry))
print("{}/{}".format(len(out), len(data)))
for i in range(min(len(out),5)):
    print(out[i])