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

data = json.load(open('../data/data.json'))
data_len = len(data)
seen_data_ratio = 0.9
seen_data_num = int(data_len * seen_data_ratio)

seen_test_data = []
unseen_test_data = []
text_num = 5

for i, [question, answer] in enumerate(data):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024)
    result = pipe(f"<s>[INST] Please give me a list of {text_num} rephrased sentence of following sentence: {question} [/INST]")[0]
    result = result['generated_text'].replace(f"<s>[INST] Please give me a list of {text_num} rephrased sentence of following sentence: {question} [/INST]", '').replace('</s>', '')
    re.sub(r'  ', ' ', result)
    rephrased_sentences = result.split('\n')

    sentence_num = 1
    for rephrased_sentence in rephrased_sentences:
        if str(sentence_num) == rephrased_sentence[:len(str(sentence_num))]:
            if i < seen_data_num:
                seen_test_data.append( {'text' : f"<s>[INST] {rephrased_sentence[len(str(sentence_num))+2:]} [/INST] {answer} </s>"} )
            else:
                unseen_test_data.append( {'text' : f"<s>[INST] {rephrased_sentence[len(str(sentence_num))+2:]} [/INST] {answer} </s>"} )
            sentence_num += 1

with open('../data/seen_test_data.json', 'w') as f : 
    json.dump(seen_test_data, f, indent=4)

with open('../data/unseen_test_data.json', 'w') as f : 
    json.dump(unseen_test_data, f, indent=4)