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
from evaluate import evaluator
from datasets import Dataset


model_name = "../models/llama-2-7b-chat-hf"

new_model = "../models/new-llama2-model-no-aug"

device_map = {"": 0}

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
)

fine_tuned_model = PeftModel.from_pretrained(base_model, new_model)
fine_tuned_model = fine_tuned_model.merge_and_unload()

# Reload tokenizer to save it
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, max_length=1024)
fine_tuned_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token
fine_tuned_tokenizer.padding_side = "right"

task_evaluator = evaluator("text-generation")

seen_test_data_list = json.load(open('../data/seen_test_data.json'))
seen_test_dataset = Dataset.from_dict({"text": [item["text"] for item in seen_test_data_list]})

unseen_test_data_list = json.load(open('../data/unseen_test_data.json'))
unseen_test_dataset = Dataset.from_dict({"text": [item["text"] for item in unseen_test_data_list]})

eval_results_for_seen_data = task_evaluator.compute(
    model_or_pipeline=fine_tuned_model,
    tokenizer=fine_tuned_tokenizer,
    data=seen_test_dataset,
    metric='bleu'
)
print("results for seen data")
print(eval_results_for_seen_data)

eval_results_for_unseen_data = task_evaluator.compute(
    model_or_pipeline=fine_tuned_model,
    tokenizer=fine_tuned_tokenizer,
    data=unseen_test_dataset,
    metric='bleu'
)
print("results for unseen data")
print(eval_results_for_unseen_data)