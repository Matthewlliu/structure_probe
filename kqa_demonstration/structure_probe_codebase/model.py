import transformers
import torch
import openai
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer
import deepspeed
import requests
import json

def get_model_by_name(model_name):
    return {
        'gpt2': GPT2LMHeadModel,
        'gpt2-large': GPT2LMHeadModel,
        'gpt2-XL': GPT2LMHeadModel,
        'gpt-j': AutoModelForCausalLM,
        'flan-t5-large': T5ForConditionalGeneration,
        'flan-t5-xl': T5ForConditionalGeneration,
        'flan-t5-xxl': T5ForConditionalGeneration
    }[model_name]

def get_tokenizer_by_name(model_name):
    return {
        'gpt2': GPT2Tokenizer,
        'gpt2-large': GPT2Tokenizer,
        'gpt2-XL': GPT2Tokenizer,
        'gpt-j': AutoTokenizer,
        'flan-t5-large': AutoTokenizer,
        'flan-t5-xl': AutoTokenizer,
        'flan-t5-xxl': AutoTokenizer
    }[model_name]

class HuggingfaceModel(object):
    def __init__(self, args):
        self.args = args
        self.get_model_and_tokenizer(args)
        self._cuda_device = 'cpu'
        self.try_cuda()

    def get_model_and_tokenizer(self, args):
        model_cls = get_model_by_name(args.model_name)
        self.model = model_cls.from_pretrained(args.model_dir)
        tokenizer_cls = get_tokenizer_by_name(args.model_name)
        self.tokenizer = tokenizer_cls.from_pretrained(args.model_dir)
        if args.model_name.startswith('gpt') or args.model_name.startswith('flan'):
            self.model = deepspeed.init_inference(
                model=self.model,      # Transformers models
                mp_size=1,        # Number of GPU
                dtype=torch.float16, # dtype of the weights (fp16)
                replace_method="auto", # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=True, # replace the model with the kernel injector
            )

    def generate_text(self, input_texts, max_length=1024, decoding='sampling', suffix='', isfilter=True):
        self.model.eval()
        sentences_list = []
        with torch.no_grad():
            kwargs = {'max_length': max_length}
            if decoding == 'sampling':
                kwargs['do_sample'] = True
                kwargs['top_k'] = self.args.topk
                kwargs['top_p'] = self.args.topp
                kwargs['temperature'] = self.args.temperature
                kwargs['num_return_sequences'] = 1 # self.args.num_generate
                kwargs['pad_token_id'] = self.tokenizer.eos_token_id
            elif decoding == 'beam_gen':
                kwargs['do_sample'] = False
                kwargs['num_beams'] = self.args.beam_size
                kwargs['temperature'] = self.args.temperature
                kwargs['num_return_sequences'] = self.args.num_generate
                kwargs['pad_token_id'] = self.tokenizer.eos_token_id
            elif decoding == 'beam_sample':
                kwargs['do_sample'] = True
                kwargs['top_k'] = self.args.topk
                kwargs['top_p'] = self.args.topp
                kwargs['num_beams'] = self.args.beam_size
                kwargs['temperature'] = self.args.temperature
                kwargs['num_return_sequences'] = 1 #self.args.num_generate
                kwargs['pad_token_id'] = self.tokenizer.eos_token_id
            for input_text in input_texts:
                sequences = []
                input_text = input_text.strip()
                input_text += suffix
                #logging.info('Start to generate from "{}"'.format(input_text))
                input_encoding = self.tokenizer.encode(
                    input_text, return_tensors='pt')
                if input_encoding.size(1)>max_length:
                    input_encoding = input_encoding[:, :max_length]
                input_encoding = input_encoding.to(self._cuda_device)
                generated_tokens = self.model.generate(
                    input_encoding, **kwargs)
                for tok_seq in generated_tokens:
                    sequence = self.tokenizer.decode(tok_seq)
                    if isfilter is True:
                        sequence = self.filter_special_tokens(sequence)
                    #print(sequence)
                    sequences.append(sequence)
                    
                sentences_list.append(sequences)
        return sentences_list
    
    def filter_special_tokens(self, sent, eos='<|endoftext|>'):
        if self.args.model_name.startswith('gpt'):
            while sent.endswith(eos):
                sent = sent[:-len(eos)].strip()
            return sent
        elif self.args.model_name.startswith('flan'):
            pass
            return sent
        else:
            raise ValueError("invalid model type %s" % self.args.model_name)
        return sent
    
    def try_cuda(self):
        if torch.cuda.is_available():
            if self._cuda_device != 'cuda':
                self._cuda()
                self._cuda_device = torch.device("cuda")
        else:
            raise ValueError("Cuda Device Not Found!")
            
    def _cuda(self):
        #self.model.to(self._cuda_device)
        self.model.cuda()
        
class GLMApi(object):
    def __init__(self, args):
        self.args = args
    
    def sending_post(self, texts, stop = [], regix = ""):
        if self.args.strategy.startswith('beam'):
            strategy = "BeamSearchStrategy"
        # If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
        data = {
            "prompt": texts,
            "max_tokens": 150,
            "min_tokens": 0,
            "top_k": self.args.topk,
            "top_p": self.args.topp,
            "temperature": self.args.temperature,
            "seed": 501,
            "num_beams": self.args.beam_size,
            "length_penalty": 0.9,
            "no_repeat_ngram_size": 3,
            "regix": regix
        }

        #t = time.time()
        #http://180.184.81.171:9721/generate
        #http://180.184.97.60:9624/generate
        res = requests.post("http://180.184.81.171:9722/generate", json=data).content.decode()
        #t = time.time() - t

        res = json.loads(res)
        res = res['text']
        res = [r[0] for r in res]
        return res
    
class OpenaiReq():
    def __init__(self, args):
        self.args = args
        self.all_keys = self.load_keys(keys = self.args.model_dir)
        self.keys_available = [True for i in range(len(self.all_keys))]
        self.key_it = 0
    
    def load_keys(self, keys):
        with open(keys, 'r') as f:
            keys = f.readlines()
        keys = [ k.strip() for k in keys ]
        print("Successfully load %s keys" % len(keys))
        return keys
    
    def req2openai(self,prompt,max_tokens=128):
        openai.api_key = self.all_keys[self.key_it]
        if not any(self.keys_available):
            print('run out of keys')
            return '', False
        data = {
            'model': self.args.model_name,
            'prompt': prompt,
            'temperature': self.args.temperature,
            'max_tokens': max_tokens,
            'top_p': self.args.topp
        }
        response = None
        while response == None:
            try:
                #response = openai.Completion.create(model=self.args.model_name, prompt=prompt, 
                #                            temperature=self.args.temperature, max_tokens=max_tokens, top_p=self.args.topp)
                response = requests.post("http://103.238.162.37:10071/completion/no_cache", json=data).content.decode()
            except Exception as e:
                err_msg = str(e)
                print(e)
                if 'server' in err_msg: # this err_msg occure when openai server is down
                    time.sleep(2) # sleep a while, then try again
                    continue
                if "reduce your prompt" in err_msg: # this is because the input string too long
                    if type(prompt) == list:
                        return ['too long' for _ in range(len(prompt))], False 
                    else:
                        return 'too long', False
                if any(self.keys_available):
                    #print("ERROR:",err_msg)
                    #print('quota' in err_msg)
                    #input()
                    if 'quota' in err_msg:
                        self.keys_available[self.key_it] = False # run out of quota
                        print(f'we have {sum(self.keys_available)} keys available')
                    # maybe reach the rate limit, switch to next key
                    self.key_it = (self.key_it + 1) % len(self.all_keys)
                    print(f'1:switch to next key {self.all_keys[self.key_it]}')
                    while (self.keys_available[self.key_it] is False) and (any(self.keys_available)): # skip keys which has run out of quota
                        self.key_it = (self.key_it + 1) % len(self.all_keys)
                        print(f'2:switch to next key {self.all_keys[self.key_it]}')
                    openai.api_key = self.all_keys[self.key_it]
                else:
                    if self.args.spare_keys is not None:
                        self.all_keys = self.load_keys(keys = self.args.spare_keys)
                        self.keys_available = [True for i in range(len(self.all_keys))]
                        self.key_it = 0
                    else:
                        print('run out of keys')
                        return '', False
        if response == None:
            return '', response
        else:
            try:
                response = json.loads(response)
            except:
                print(response)
                exit()
            if type(prompt) == list:
                return [str(v['text']) for v in response['choices']], response
            else:
                return response['choices'][0]['text'], response
    
    def req2chatgpt(self,prompt,model='gpt-3.5-turbo'):
        openai.api_key = self.all_keys[self.key_it]
        if not any(self.keys_available):
            print('run out of keys')
            return '', False
        response = None
        max_retry = 20

        try_num = 0
        while(response == None):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ]
                    )
            except Exception as e:
                err_msg = str(e)
                print(e)
                if 'server' in err_msg: # this err_msg occure when openai server is down
                    time.sleep(2) # sleep a while, then try again
                    continue
                if "reduce your prompt" in err_msg: # this is because the input string too long
                    if type(prompt) == list:
                        return ['too long' for _ in range(len(prompt))], False 
                    else:
                        return 'too long', False
                if any(self.keys_available):
                    if 'quota' in err_msg:
                        self.keys_available[self.key_it] = False # run out of quota
                        print(f'we have {sum(self.keys_available)} keys available')
                    # maybe reach the rate limit, switch to next key    
                    self.key_it = (self.key_it + 1) % len(self.all_keys)
                    print(f'switch to next key {self.all_keys[self.key_it]}')
                    while not self.keys_available[self.key_it]: # skip keys which has run out of quota
                        self.key_it = (self.key_it + 1) % len(self.all_keys)
                        print(f'switch to next key {self.all_keys[self.key_it]}')
                    openai.api_key = self.all_keys[self.key_it]
                else:
                    print('run out of keys')
                    return '', False
                try_num += 1
                if try_num >= max_retry:
                    print("%s api error, exiting" % self.args.model_name)
                    exit()
        if response == None:
            return '', response
        else:
            if type(prompt) == list:
                return [str(v['text']) for v in response['message']['content']], response
            else:
                return response['choices'][0]['message']['content'], response

def get_model_from_local(args):
    if args.model_name in ['gpt2', 'gpt2-large', 'gpt2-XL', 'gpt-j', 'flan-t5-large', 'flan-t5-xl', 'flan-t5-xxl']:
        return HuggingfaceModel(args)
    else:
        raise ValueError("Invalid model name: %s" % args.model_name)
        
def get_model_api(args):
    if args.model_name in ['text-davinci-001', 'text-davinci-003', 'code-davinci-002', 'chatgpt', 'GPT4']:
        return OpenaiReq(args)
    elif args.model_name in ['glm-130b']:
        return GLMApi(args)

if __name__=='__main__':
    texts = ["Who was the first emperor in France, and where was him from?",
        "The Starry Night is an oil-on-canvas painting by [MASK] in June 1889."]
    '''
    data = {
            "prompt": texts,
            "max_tokens": 640,
            "min_tokens": 0,
            "top_k": 1,
            "top_p": 0,
            "temperature": 1.0,
            "seed": 501,
            "num_beams": 5,
            "length_penalty": 0.9,
            "no_repeat_ngram_size": 3,
            "regix": ""
        }
    res = requests.post("http://180.184.81.171:9722/generate", json=data).content.decode()
    if res=='':
        print("result is empty")
    #res = json.loads(res)
    print(res)
    '''
    data = {
        'model': 'GPT-4',
        'prompt': texts,
        'temperature': 1,
        'max_tokens': 128,
        'top_p': 1.0
    }
    #response = openai.Completion.create(model='text-davinci-003', prompt=texts, 
    #                                    temperature=1, max_tokens=128)
    response = requests.post("http://103.238.162.37:10071/chat_completion/no_cache", json=data).content.decode()
    print(response)
    response = json.loads(response)
    ans = [str(v['text']).strip() for v in response['choices']]
    print(ans)