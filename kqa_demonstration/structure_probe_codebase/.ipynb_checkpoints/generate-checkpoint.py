import os
import json
from tqdm import tqdm
from utils import post_process, post_process_api

def generate_from_online(post_api, dataset, args):
    aug_part = dataset.data[args.start_id:args.augment_size]
    seq_list = []
    ind = 0
    indexs = list(range(len(aug_part)))
    
    #total = len(aug_part)//args.batch_size + 1
    pbar = tqdm(total=len(aug_part))
    
    while(ind*args.batch_size <= len(aug_part)):
        s = slice(ind*args.batch_size, (ind+1)*args.batch_size)
        mini_batch = aug_part[s]
        index = indexs[s]
        
        prompts = []
        for i, entry in zip(index, mini_batch):
            prompt = dataset.retrieve_demostrations(entry, i)
            prompts.append(prompt)
        if args.model_name in ['chatgpt']:
            sequences, _ = post_api.req2chatgpt(prompts)
        else:
            sequences, _ = post_api.req2openai(prompts, model=args.model_name, 
                                        temperature=args.temperature, top_p=args.topp)
        if _ is False:
            raise ValueError("Run out all keys")
        sequences = [post_process_api( s.strip() ) for s in sequences]
        seq_list.extend(sequences)
            
        if (ind + 1)*args.batch_size % args.save_step == 0 or (ind + 1)*args.batch_size>=len(aug_part):
            save_id = (ind*args.batch_size)//args.save_step
            s_id = save_id * args.save_step
            e_id = min((save_id + 1) * args.save_step, len(aug_part))
            save_data(aug_part[s_id:e_id], seq_list[s_id:e_id], args, save_id + args.start_id//args.save_step)
        pbar.update(args.batch_size)
        ind += 1
        

def generate_from_local_model(model, dataset, args):
    aug_part = dataset.data[:args.augment_size]
    seq_list = []
    for ind, entry in enumerate(tqdm(aug_part)):
        prompt = dataset.retrieve_demostrations(entry, ind)
        #print("Golden: {}".format(entry['question']))
        #print("Prompt: {}".format(prompt))
        sequence = model.generate_text([prompt], decoding='sampling')[0][0]
        #print(sequence)
        sequence = post_process(sequence, args.demo_num)
        #print("Generated text: {} \n".format(sequence))
        seq_list.append(sequence)
        
        if (ind + 1) % args.save_step == 0 or (ind + 1)==len(aug_part):
            save_id = ind//args.save_step
            s_id = save_id * args.save_step
            e_id = min((save_id + 1) * args.save_step, len(aug_part))
            save_data(aug_part[s_id:e_id], seq_list[s_id:e_id], args, save_id)

def save_data(aug_part, seq_list, args, save_id):
    for entry, aug_que in zip(aug_part, seq_list):
        entry['question'] = aug_que
    with open(os.path.join(args.output_dir, 'generated_ques_%s.json' % save_id), 'w') as f:
        json.dump(aug_part, f)