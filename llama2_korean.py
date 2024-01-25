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


model_name = "beomi/llama-2-ko-7b"#"KRAFTON/KORani-v3-13B"
device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

while True:
    question = input('question: ')
    result = pipe(f"[INST] {question} [/INST]")[0]['generated_text']
    result = re.sub(r' +', ' ', result)
    result = re.sub(r'\s{2,}', '\n', result)
    print()
    print(result)
    print()
