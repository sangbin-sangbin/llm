from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
import json


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
unmasker = pipeline('fill-mask', model='bert-base-cased')

def random_replace(_input_text, n):
    def helper(input_text):
        input_text_list = input_text.split()
        len_input = len(input_text_list)
        
        rand_idx = random.randint(1,len_input-1)
        orig_word = input_text_list[rand_idx]
        new_text_list = input_text_list.copy()
        new_text_list[rand_idx] = tokenizer.mask_token
        new_mask_sent = ' '.join(new_text_list)

        augmented_text_list = unmasker(new_mask_sent)
        for res in augmented_text_list:
            if res['token_str'] != orig_word:
                augmented_text = res['sequence']
                break
        return augmented_text

    res = _input_text
    for _ in range(n):
        res = helper(res)
    return res

data = json.load(open('data/data.json'))
augmented_data = []

res = input('data augmentation? [y / n]')

if res == 'n':
    for q, a in data:
        augmented_data.append( {'text' : f"<s>[INST] {question} [/INST] {answer} </s>"} )
    with open('data/not_augmented_data.json', 'w') as f : 
        json.dump(augmented_data, f, indent=4)
else:
    for q, a in data:
        augmented_data.append( {'text' : f"<s>[INST] {question} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 1)} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 2)} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 3)} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 4)} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 5)} [/INST] {answer} </s>"} )
    with open('data/augmented_data.json', 'w') as f : 
        json.dump(augmented_data, f, indent=4)


    