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

pipe = pipeline("text-generation", model="KRAFTON/KORani-v3-13B")

while True:
    question = input('question: ')
    result = pipe(f"# {question}\n### Assistant: 1)")[0]['generated_text']
    result = re.sub(r' +', ' ', result)
    result = re.sub(r'\s{2,}', '\n', result)
    print()
    print(result)
    print()
