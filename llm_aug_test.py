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

while True:
    prompt = input('prompt: ')
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024)
    result = pipe(f"<s>[INST] Please give me a list of 10 rephrased sentence of following sentence: {prompt} [/INST]")[0]
    result = result['generated_text'].replace(f"<s>[INST] Please rephrase the following sentence in json format: {prompt} [/INST]", '').replace('</s>', '')
    re.sub(r'\s', '', result)
    rephrased_sentences = result.split('\n')

    sentence_num = 1
    for rephrased_sentence in rephrased_sentences:
        if str(sentence_num) == rephrased_sentence[:len(str(sentence_num))]:
            print()
            print(rephrased_sentence[len(str(sentence_num))+2:])
            print()
            sentence_num += 1
