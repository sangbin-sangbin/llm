from trlx.data.default_configs import default_ppo_config
import trlx
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

config = default_ppo_config()
config.model = fine_tuned_model
config.tokenizer = fine_tuned_tokenizer
config.train.seq_length = 16


question = input('question: ')
pipe = pipeline(task="text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, max_length=1024)
result = pipe(f"<s>[INST] {question} [/INST]")[0]['generated_text'].replace(f"<s>[INST] {question} [/INST]", '').replace('</s>', '')
re.sub(r' +', ' ', result)
re.sub(r'\s{2,}', '\n', result)

print(f'\n{result}\n')

question = input('question: ')
classification_model = AutoModelForSequenceClassification.from_pretrained("../models/classification")
classification_tokenizer = AutoTokenizer.from_pretrained("../models/classification")
classify = pipeline(task="sequence-classification", model=classification_model, tokenizer=classification_tokenizer)


reward = gpt2_for_reward(question)

trainer = trlx.train(config=config, samples=[question], rewards=[reward])
trainer.save_pretrained('../models/rlhf')
