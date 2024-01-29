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
data_len = len(data)
seen_data_ratio = 0.9
seen_data_num = int(data_len * seen_data_ratio)
augmented_data = []

aug_type = input('what augmentation? [no / bert / llm]\n>>> ')

if aug_type == 'no':
    for i, [question, answer] in enumerate(data):
        if i == seen_data_num: break
        augmented_data.append( {'text' : f"<s>[INST] {question} [/INST] {answer} </s>"} )
    print(len(augmented_data), "data created")
    with open('../data/no_augmented_data.json', 'w') as f :
        json.dump(augmented_data, f, indent=4)
elif aug_type == 'bert':
    for i, [question, answer] in enumerate(data):
        if i == seen_data_num: break
        augmented_data.append( {'text' : f"<s>[INST] {question} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 1)} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 2)} [/INST] {answer} </s>"} )
        augmented_data.append( {'text' : f"<s>[INST] {random_replace(question, 3)} [/INST] {answer} </s>"} )
    print(len(augmented_data), "data created")
    with open('../data/bert_augmented_data.json', 'w') as f:
        json.dump(augmented_data, f, indent=4)
else:
    model_name = "../models/llama-2-7b-chat-hf"

    # Load the entire model on the GPU 0
    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logging.set_verbosity(logging.CRITICAL)

    text_num = 10

    for i, [question, answer] in enumerate(data):
        if i == seen_data_num: break
        augmented_data.append( {'text' : f"<s>[INST] {question} [/INST] {answer} </s>"} )

        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024)
        result = pipe(f"<s>[INST] Please give me a list of {text_num} rephrased sentence of following sentence: {question} [/INST]")[0]
        result = result['generated_text'].replace(f"<s>[INST] Please give me a list of {text_num} rephrased sentence of following sentence: {question} [/INST]", '').replace('</s>', '')
        re.sub(r'  ', ' ', result)
        rephrased_sentences = result.split('\n')

        sentence_num = 1
        for rephrased_sentence in rephrased_sentences:
            if str(sentence_num) == rephrased_sentence[:len(str(sentence_num))]:
                augmented_data.append( {'text' : f"<s>[INST] {rephrased_sentence[len(str(sentence_num))+2:]} [/INST] {answer} </s>"} )
                sentence_num += 1
        if sentence_num - 1 != text_num:
            print("\nERROR!!!", sentence_num - 1, "text generated.\n")
    print(len(augmented_data), "data created")
    with open('../data/llm_augmented_data.json', 'w') as f : 
        json.dump(augmented_data, f, indent=4)