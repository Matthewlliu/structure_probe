import os
import json
from tqdm import tqdm
from utils import post_process, post_process_api

sample_path = '/home/ljx/structure_probe/KQAPro_Baselines-master/sampled_grailqa_dev_120.json'
with open(sample_path, 'r') as f:
    sample_id = json.load(f)

def generate_from_online(post_api, dataset, args):
    if args.if_lf2nl:
        aug_part = dataset.aug_part[args.start_id:args.augment_size]
    else:
        if args.logic_forms == 'lambdaDCS':
            aug_part = []
            for k, inds in sample_id.items():
                with open(os.path.join(args.test_dir, k+'_test.tsv'), 'r') as f:
                    tmp = f.readlines()
                    aug_part.extend([ tmp[i] for i in inds ])
        else:
            with open(args.test_dir, 'r') as f:
                aug_part = json.load(f)
                #aug_part = aug_part[args.start_id:args.augment_size]
                aug_part = [  aug_part[i] for i in sample_id ]

        if args.toy:
            aug_part = aug_part[:5]
        print("Length of test set: {}".format(len(aug_part)))
    seq_list = []
    ind = 0
    indexs = list(range(len(aug_part)))
    
    #total = len(aug_part)//args.batch_size + 1
    pbar = tqdm(total=len(aug_part))
    
    while(ind*args.batch_size < len(aug_part)):
        s = slice(ind*args.batch_size, (ind+1)*args.batch_size)
        mini_batch = aug_part[s]
        index = indexs[s]
        
        prompts = []
        entities = []
        for i, entry in zip(index, mini_batch):
            prompt, entity = dataset.retrieve_demonstrations(entry, i+args.start_id)
            prompts.append(prompt)
            if entity is not None:
                entities.append(entity)
            #print("Program: ",entry['lambda-dcs'])
            #print("Prompt:, ",prompt)
        if args.model_name in ['glm-130b']:
            sequences = post_api.sending_post(prompts)
            _ = True
        elif args.model_name in ['chatgpt']:
            sequences, _ = post_api.req2chatgpt(prompts)
        else:
            sequences, _ = post_api.req2openai(prompts)
        if _ is False:
            raise ValueError("Run out all keys")
        
        #print(sequences)
        sequences = [post_process_api( s.strip() ) for s in sequences]
        if len(entities) > 0:
            sequences = [ s for s in zip(sequences, entities)]
        #print("generated: ", sequences)
        #input()
        seq_list.extend(sequences)
            
        if (ind + 1)*args.batch_size % args.save_step == 0 or (ind + 1)*args.batch_size>=len(aug_part):
            save_id = (ind*args.batch_size)//args.save_step
            s_id = save_id * args.save_step
            e_id = min((save_id + 1) * args.save_step, len(aug_part))
            save_data(aug_part[s_id:e_id], seq_list[s_id:e_id], args, save_id + args.start_id//args.save_step)
        pbar.update(args.batch_size)
        ind += 1
        

def generate_from_local_model(model, dataset, args):
    if args.if_lf2nl:
        aug_part = dataset.aug_part[args.start_id:args.augment_size]
    else:
        if args.logic_forms == 'lambdaDCS':
            aug_part = []
            for k, inds in sample_id.items():
                with open(os.path.join(args.test_dir, k+'_test.tsv'), 'r') as f:
                    tmp = f.readlines()
                    aug_part.extend([ tmp[i] for i in inds ])
        else:
            with open(args.test_dir, 'r') as f:
                aug_part = json.load(f)
                #aug_part = aug_part[args.start_id:args.augment_size]
                aug_part = [  aug_part[i] for i in sample_id ]

        if args.toy:
            aug_part = aug_part[:5]
        print("Length of test set: {}".format(len(aug_part)))
        
    seq_list = []
    for ind, entry in enumerate(tqdm(aug_part)):
        prompt, entity = dataset.retrieve_demonstrations(entry, ind+args.start_id)
        #print("Golden: {}".format(entry['question']))
        #print("Prompt: {}".format(prompt))
        #try:
        sequence = model.generate_text([prompt], decoding='beam_sample')[0][0]
        #except RuntimeError:
        #    print("Golden: {}".format(entry['question']))
        #    print("Prompt: {}".format(prompt))
        #    continue
        #print(sequence)
        sequence = post_process(sequence, args.demo_num)
        if entity is not None:
            sequence = [sequence, entity]
        #print("Generated text: {} \n".format(sequence))
        #input()
        seq_list.append(sequence)
        
        if (ind + 1) % args.save_step == 0 or (ind + 1)==len(aug_part):
            save_id = ind//args.save_step
            s_id = save_id * args.save_step
            e_id = min((save_id + 1) * args.save_step, len(aug_part))
            save_data(aug_part[s_id:e_id], seq_list[s_id:e_id], args, save_id + args.start_id//args.save_step)

def save_data(aug_part, seq_list, args, save_id):
    if args.logic_forms in ['kopl', 'lambdaDCS_kqapro']:
        for entry, seq in zip(aug_part, seq_list):
            if args.if_lf2nl:
                entry['question'] = seq
            else:
                entry['pred'] = seq
    elif args.logic_forms == 'lambdaDCS':
        for ind, seq in enumerate(seq_list):
            aug_part[ind] = seq  + '\t' + aug_part[ind].split('\t')[1]
    elif args.logic_forms in ['sparql', 'sparql_kqapro']:
        for entry, seq in zip(aug_part, seq_list):
            if args.if_lf2nl:
                entry['question'] = seq
            else:
                entry['pred'] = seq[0]
                entry['entities'] = seq[1]
    else:
        raise ValueError("Not implemented type %s for saving" % args.logic_forms)

    if args.if_lf2nl:
        file_name = 'generated_ques_%s.json' % save_id
    else:
        file_name = 'pred.json'

    with open(os.path.join(args.output_dir, file_name), 'w') as f:
        json.dump(aug_part, f)