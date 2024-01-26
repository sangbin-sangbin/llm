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


model_name = "../models/llama-2-7b-chat-hf"
new_model = '../models/new-llama2-model-llama-aug'
device_map = {"": 0}

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float32,
    device_map='cpu',
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

test_strings = [
    "Which Remote Services can I use for my vehicle in conjunction with the My BMW App?",
    "Who are you?",
    "hello, nice to meet you.",
    "How can I buy BMW?",
    "How much is tesla model Y?",
    "I want to go home.",
    "I will find you and kill you",
    "hello my name is sangbin",
    "please call 911 I am dying",
    "how do you think about stock of BMW, should I buy it?",
    "what is lotto number of next week?",
    "Is a map update available for my BMW?",
    "How can I activate or deactivate the vehicle tracking of my BMW?",
    "How can I restart or \"reboot\" the BMW iDrive system in my vehicle?",
    "How do I track the status of my new BMW vehicle order online?",
    "How do I activate the \"Remote Engine Start\" function for my BMW?",
    "How do I log in or out of Spotify via BMW Connected Music in my BMW?",
    "Which login data can I use to access the My BMW App or the My BMW portal on the web?",
    "How do I activate the Key Card for my BMW Digital Key?",
    "How do I add a vehicle to my BMW ID in the My BMW App?",
    "How can I delete my personal data and settings in my BMW?",
    "How do I see the last parking position of my BMW?",
    "Why does my BMW activate the guest profile, even though I have already created a driver profile with my BMW ID?",
    "How can I air-condition my BMW with the Remote Services of the My BMW App?",
    "How can I find detailed information on my last trip in the My BMW App?",
    "How do I delete all stored destinations in the navigation system of my BMW?",
    "How do I remove a vehicle from the My BMW App?",
    "How do I create a BMW ID?",
    "Where can I change the account data for my BMW ID?",
    "Does the My BMW App or the My BMW portal show me the electronic service history of my BMW?",
]
without_vino_total_time = 0
for test_string in test_strings:
    input_tokens = fine_tuned_tokenizer(f"<s>[INST] {test_string} [/INST] ", return_tensors="pt", add_special_tokens=False)
    start = time.time()
    result = pipe(f"<s>[INST] {test_string} [/INST]")
    end = time.time()
    result = result[0]['generated_text'].replace(f"<s>[INST] {test_string} [/INST]", '').replace('</s>', '')
    result = re.sub(r' +', ' ', result)
    result = re.sub(r'\s{2,}', '\n', result)
    print(result)
    without_vino_total_time += end - start






from transformers import AutoTokenizer, AutoConfig
import openvino as ov
from pathlib import Path
from optimum.intel.openvino import OVModelForCausalLM
import time


base_model = '../models/llama-2-7b-chat-hf'
model_dir = Path('../models/llama2_vino')

core = ov.Core()
device = "CPU"
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device=device,
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
)

with_vino_total_time = 0
for test_string in test_strings:
    input_tokens = tokenizer(f"<s>[INST] {test_string} [/INST] ", return_tensors="pt", add_special_tokens=False)
    start = time.time()
    result = ov_model.generate(**input_tokens, max_new_tokens=1024)
    end = time.time()
    print(tokenizer.batch_decode(result, skip_special_tokens=True)[0])
    with_vino_total_time += end - start
    
print("\n\nwith_vino_total_time:", with_vino_total_time, "\n\nwithout_vino_total_timee:", without_vino_total_time)
