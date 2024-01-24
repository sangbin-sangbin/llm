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


classification_model = AutoModelForSequenceClassification.from_pretrained("../models/classification")
classification_tokenizer = AutoTokenizer.from_pretrained("../models/classification")
classify = pipeline(task="sequence-classification", model=classification_model, tokenizer=classification_tokenizer)

print(classify("I hate you"))
print(classify("I love you"))
print(classify("fuck you"))
print(classify("i will find you and kill you"))