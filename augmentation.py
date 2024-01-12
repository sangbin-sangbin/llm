from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
import json
import os
from transformers import (
    AutoModelForCausalLM,
    logging,
)
from peft import PeftModel
import re

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
unmasker = pipeline('fill-mask', model='bert-base-cased')

def random_replace(_input_text, n):
    def helper(input_text):
        input_text_list = input_text.split()
        len_input = len(input_text_list)
        if len_input <= 1:
            return input_text

        rand_idx = random.randint(0,len_input-1)
        orig_word = input_text_list[rand_idx]
        new_text_list = input_text_list.copy()
        new_text_list[rand_idx] = tokenizer.mask_token
        new_mask_sent = ' '.join(new_text_list)

        augmented_text_list = unmasker(new_mask_sent)
        augmented_text = input_text
        for res in augmented_text_list:
            if res['token_str'] != orig_word and len(res['token_str']) > 0:
                augmented_text = res['sequence']
                break
        return augmented_text

    res = _input_text
    for _ in range(n):
        res = helper(res)
    return res

data = json.load(open('../data/data.json'))
augmented_data = []

res = input('what augmentation? [no / bert / llm]\n>>> ')

if res == 'no':
    for question, answer in data:
        augmented_data.append( {'text' : f"<s>[INST] {question} [/INST] {answer} </s>"} )
elif res == 'bert':
    for question, answer in data:
        augmented_data.append( {'text' : f"<s>[INST] {question} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 1)} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 2)} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 3)} [/INST] {answer} </s>"} )
else:
    for question, answer in data:
        augmented_data.append( {'text' : f"<s>[INST] {question} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {question} [/INST] {answer} </s>"} )

with open('../data/augmented_data.json', 'w') as f : 
    json.dump(augmented_data, f, indent=4)
    

