import os
import numpy as np
import torch
import random

def seed_everything(seed=501):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    
def ensemble_input(demon, inputs, lf, reverse=True):
    prompt = 'According to the given logic form %s, generate the corresponding natural language question. ' % lf
    if reverse:
        demon = reversed(demon)
    prompt += 'For examples, '
    for pair in demon:
        prompt += pair[1] + ' is verbalized as: ' + pair[0] + ' [SEP] '
    prompt += inputs + ' is verbalized as: '
    return prompt

def levenshtein(l1, l2, thresh=np.inf):
    """
        l1, l2: list of kopl functions
        thresh: maximum edit distance allowed minus 1
    """
    len1 = len(l1)
    len2 = len(l2)
    dp = np.zeros([len1+1, len2+1])
    for i in range(1, len1+1):
        dp[i][0] = i
    for j in range(1, len2+1):
        dp[0][j] = j
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if l1[i-1] == l2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = np.min([dp[i-1][j-1], dp[i-1][j], dp[i][j-1]]) + 1
            if dp[i][j] > thresh + 1:
                return dp[i][j]
    return int(dp[len1][len2])

def post_process(seq, demo_num):
    gen = seq.split('is verbalized as:')
    if len(gen)>=demo_num+2:
        gen = gen[demo_num+1]
    else:
        gen = gen[-1]
    ans = gen.split('[SEP]')[0]

    # for t5
    ans = ans.replace('<pad>', '').replace('</s>', '')
    return ans

def post_process_api(seq):
    gen = seq.split('[SEP]')[0]
    gen = gen.strip()
    while(gen[0]=='\n'):
        gen = gen[1:]
    return gen

if __name__=='__main__':
    pass