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
import evaluate
from datasets import Dataset
from tqdm import tqdm


model_name = "../models/llama-2-7b-chat-hf"

new_model = "../models/new-llama2-model-no-aug"

device_map = {"": 0}

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map
)

fine_tuned_model = PeftModel.from_pretrained(base_model, new_model)
fine_tuned_model = fine_tuned_model.merge_and_unload()

# Reload tokenizer to save it
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, max_length=1024)
fine_tuned_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token
fine_tuned_tokenizer.padding_side = "right"

seen_test_data_list = json.load(open('../data/seen_test_data.json'))
unseen_test_data_list = json.load(open('../data/unseen_test_data.json'))

pipe = pipeline(task="text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer, max_length=1024)
    
logging.set_verbosity(logging.CRITICAL)

def bleu_evaluation(dataset):   
    dataset = dataset[:8]
    predictions = [] 
    references = []
    batch_size = 8
    l = len(dataset)
    with tqdm(total=l, desc="Processing", unit="item") as pbar:
        batch = []
        for i in range(l):
            txt = dataset[i]['text']
            idx = txt.find('[/INST]')
            q = txt[:idx+7]
            a = txt[idx+7:]
            batch.append(q)
            references.append([a])
            pbar.update(1)
            if len(batch) == batch_size:
                res = pipe(batch)
                for i, p in enumerate(res):
                    predictions.append(p[0]['generated_text'].replace(batch[i], '').replace('</s>', ''))
                batch = []
        if len(batch) > 0:
            res = pipe(batch)
            for i, p in enumerate(res):
                predictions.append(p[0]['generated_text'].replace(batch[i], '').replace('</s>', ''))
            batch = []


    print(predictions)
    print()
    print(references)
    print()
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    return results

eval_results_for_seen_data = bleu_evaluation(seen_test_data_list)
print("results for seen data")
print(eval_results_for_seen_data)

eval_results_for_unseen_data = bleu_evaluation(unseen_test_data_list)
print("results for unseen data")
print(eval_results_for_unseen_data)


