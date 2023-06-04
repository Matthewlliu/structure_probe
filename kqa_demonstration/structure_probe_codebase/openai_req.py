import openai
import time
class OpenaiReq():
    def __init__(self,all_keys:list):
        self.all_keys = all_keys
        self.keys_available = [True for i in range(len(all_keys))]
        self.key_it = 0
    
    def req2openai(self,prompt,model="text-davinci-003",temperature=0.7,max_tokens=512, top_p=0.9):
        openai.api_key = self.all_keys[self.key_it]
        if not any(self.keys_available):
            print('run out of keys')
            return '', False
        response = None
        while response == None:
            try:
                response = openai.Completion.create(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
            except Exception as e:
                err_msg = str(e)
                print(e)
                if 'server' in err_msg: # this err_msg occure when openai server is down
                    time.sleep(5) # sleep a while, then try again
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
        if response == None:
            return '', response
        else:
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
        while response == None:
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
                    time.sleep(5) # sleep a while, then try again
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
        if response == None:
            return '', response
        else:
            if type(prompt) == list:
                return [str(v['text']) for v in response['message']['content']], response
            else:
                return response['choices'][0]['message']['content'], response
