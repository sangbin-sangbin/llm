import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)
from peft import PeftModel
import json
import re
import time


model_name = "../../models/llama-2-7b-chat-hf"
new_model = '../../models/new-llama2-model-llama-aug'
device_map = {"": 0}

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map='cuda',
)

fine_tuned_model = PeftModel.from_pretrained(base_model, new_model)
fine_tuned_model = fine_tuned_model.merge_and_unload()

# Reload tokenizer to save it
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
fine_tuned_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token
fine_tuned_tokenizer.padding_side = "right"

logging.set_verbosity(logging.CRITICAL)

pipe = pipeline(task="text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, max_length=1024)

res = {'instruction':[], 'chosen_response':[], 'rejected_response':[]}
dataset = json.load(open('../../data/data.json'))
for data in dataset:
    question = data[0]
    result1 = pipe(f"<s>[INST] {question} [/INST]")[0]['generated_text'].replace(f"<s>[INST] {question} [/INST]", '').replace('</s>', '')
    result1 = re.sub(r' +', ' ', result1)
    result1 = re.sub(r'\s{2,}', '\n', result1)
    result2 = pipe(f"<s>[INST] hello, {question} [/INST]")[0]['generated_text'].replace(f"<s>[INST] hello, {question} [/INST]", '').replace('</s>', '')
    result2 = re.sub(r' +', ' ', result2)
    result2 = re.sub(r'\s{2,}', '\n', result2)
    print(f'result1\n>>> {result1}')
    print(f'result2\n>>> {result2}')

    feedback = '.'
    while feedback not in ['0','1','2']:
        feedback = input('which one is better? [ 1 / 2 ]\n>>> ')

    if feedback == '1':
        res['instruction'].append(question)
        res['chosen_response'].append(result1)
        res['rejected_response'].append(result2)
    elif feedback == '2':
        res['instruction'].append(question)
        res['chosen_response'].append(result2)
        res['rejected_response'].append(result1)
    elif feedback == '0':
        break

