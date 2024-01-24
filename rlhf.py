from trlx.data.default_configs import default_ppo_config
import trlx
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
    AutoModelForSequenceClassification,
    TFAutoModelForSequenceClassification,
    AutoConfig
)
from peft import PeftModel
import json
import re
import time
import numpy as np
from scipy.special import softmax


model_name = "../models/llama-2-7b-chat-hf"

new_model = '../models/new-llama2-model-no-aug'

device_map = {"": 0}

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

fine_tuned_model = PeftModel.from_pretrained(base_model, new_model)
fine_tuned_model = fine_tuned_model.merge_and_unload()

fine_tuned_tokenizer = AutoTokenizer.from_pretrained(new_model, trust_remote_code=True)


rlhf_dir = '../models/rlhf'
fine_tuned_model.save_pretrained(rlhf_dir)
fine_tuned_tokenizer.save_pretrained(rlhf_dir)

del base_model
del fine_tuned_model
del fine_tuned_tokenizer

rlhf_model = AutoModelForCausalLM.from_pretrained(
    rlhf_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
rlhf_tokenizer = AutoTokenizer.from_pretrained(rlhf_dir, trust_remote_code=True)
rlhf_pipe = pipeline(task="text-generation", model=rlhf_model, tokenizer=rlhf_tokenizer, max_length=1024)

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
sent_model = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
sent_tokenizer = AutoTokenizer.from_pretrained(sent_model)
sent_config = AutoConfig.from_pretrained(sent_model)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model)

ppo_config = default_ppo_config()
ppo_config.model = rlhf_model
ppo_config.tokenizer = rlhf_tokenizer
ppo_config.train.seq_length = 16

question = input('question: ')

result = pipe(f"<s>[INST] {question} [/INST]")[0]['generated_text'].replace(f"<s>[INST] {question} [/INST]", '').replace('</s>', '')
re.sub(r' +', ' ', result)
re.sub(r'\s{2,}', '\n', result)
print(f'\n{result}\n')

while True:
    prev_question = question
    question = input('question: ')
    
    result = pipe(f"<s>[INST] {question} [/INST]")[0]['generated_text'].replace(f"<s>[INST] {question} [/INST]", '').replace('</s>', '')
    re.sub(r' +', ' ', result)
    re.sub(r'\s{2,}', '\n', result)
    print(f'\n{result}\n')

    encoded_input = sent_tokenizer(preprocess(question), return_tensors='pt')
    output = sent_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    reward = scores[2] - scores[0]

    trainer = trlx.train(config=config, samples=[prev_question], rewards=[reward])
    trainer.save_pretrained(rlhf_dir)

    del rlhf_model
    
    rlhf_model = AutoModelForCausalLM.from_pretrained(
        rlhf_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
        
