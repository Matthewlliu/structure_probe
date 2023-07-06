import os
import json
from Bart_Program.data import DataLoader
from Bart_Program.executor_rule import RuleExecutor
from tqdm import tqdm

def validate(root):
    data_root = '/home/ljx/new_cache_server32_0411/KQAPro/dataset/bart-base'
    vocab_json = os.path.join(data_root, 'vocab.json')
    val_pt = os.path.join(data_root, 'test.pt')
    val_loader = DataLoader(vocab_json, val_pt, 256)
    vocab = val_loader.vocab #train_loader.vocab
    executor = RuleExecutor(vocab, os.path.join('/home/ljx/new_cache_server32_0411/KQAPro/dataset', 'kb.json'))

    #root = '/data/ljx/result/probeLLM/kopl/nl2lf_kopl_glm-130b_2023-06-08_toy/'
    input_dir = os.path.join(root, 'pred.json')
    with open(input_dir, 'r') as f:
        data = json.load(f)

    outputs = [ d['pred'] for d in data]
    ans_choice = [ d['choices'] for d in data]

    with open(os.path.join(root, 'ans.txt'), 'w') as f:
        for output in tqdm(outputs):
            chunks = output.split('[func]')
            func_list = []
            inputs_list = []
            for chunk in chunks:
                chunk = chunk.strip()
                res = chunk.split('[arg]')
                res = [_.strip() for _ in res]
                if len(res) > 0:
                    func = res[0]
                    inputs = []
                    if len(res) > 1:
                        for x in res[1:]:
                            inputs.append(x)
                    else:
                        inputs = []
                    func_list.append(func)
                    inputs_list.append(inputs)
            ans = executor.forward(func_list, inputs_list, ignore_error = True)
            if ans == None:
                ans = 'no'
            f.write(ans + '\n')

    with open(os.path.join(root, 'ans.txt'), 'r') as f:
        ans = f.readlines()
    assert len(ans) == len(ans_choice)
    count = 0
    for i in range(len(ans)):
        if ans[i].strip() in ans_choice[i]:
            count += 1
    print("Accuracy: {} ({}/{})".format(count/len(ans), count, len(ans)))

def function_check(root):
    pred_file = os.path.join(root, 'pred.json')
    with open(pred_file, 'r') as f:
        preds = json.load(f)
    em_score = 0
    for entry in preds:
        pred = entry['pred']
        #pred_f = [ f.strip() for f in pred.split('[func]') if len(f.strip())>0]
        pred_f = remove_func_arg(pred)
        gold_f = [ f['function'] for f in entry['program']]
        if pred_f == gold_f:
            em_score += 1
            #print(entry)
    em_score /= len(preds)
    print("exact match: ", em_score)

def remove_func_arg(func):
    funcs = func.split('[func]')
    out = []
    for part in funcs:
        out.append(part.split('[arg]')[0].strip())
    return out

if __name__=='__main__':
    path = '/data/ljx/result/probeLLM/kopl/nl2lf_kopl_text-davinci-003_2023-06-27_demo35_300'
    function_check(path)
    validate(path)