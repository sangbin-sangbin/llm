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


# The model that you want to train from the Hugging Face hub
model_name = "Upstage/SOLAR-10.7B-Instruct-v1.0"

# Fine-tuned model name
new_model = "../models/new-solar-model"

# Load the entire model on the GPU 0
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

# Reload tokenizer to save it
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
fine_tuned_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token
fine_tuned_tokenizer.padding_side = "right"

logging.set_verbosity(logging.CRITICAL)

while True:
    prompt = input('question: ')
    pipe = pipeline(task="text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, max_length=1024)
    result = pipe(f"<s>[INST] {prompt} [/INST]")[0]['generated_text'].replace(f"<s>[INST] {prompt} [/INST]", '').replace('</s>', '')
    re.sub(r' +', ' ', result)
    re.sub(r'\s{2,}', '\n', result)

    print()
    print(result)
    print()