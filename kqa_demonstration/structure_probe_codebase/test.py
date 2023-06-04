"""import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed

# Model Repository on huggingface.co
model_id = "/data/ljx/cpt/gpt-j-sharded-fp16"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)


# init deepspeed inference engine
ds_model = deepspeed.init_inference(
    model=model,      # Transformers models
    mp_size=1,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True, # replace the model with the kernel injector
)
print(f"model is loaded on device {ds_model.module.device}")

# Test model
example = "My name is Philipp and I"
input_ids = tokenizer(example,return_tensors="pt").input_ids.to(model.device)
logits = ds_model.generate(input_ids, do_sample=True, max_length=100)
output = tokenizer.decode(logits[0].tolist())
"""

import os
print(os.path.isdir("cache/lambdaDCS/dataset"))