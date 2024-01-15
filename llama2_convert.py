from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel
import openvino as ov

# The model that you want to train from the Hugging Face hub
model_name = "../models/llama-2-7b-chat-hf"

# Fine-tuned model name
new_model = "../models/new-model"

device_map = {"": 0}

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
    torchscript=True
)

fine_tuned_model = PeftModel.from_pretrained(base_model, new_model)
fine_tuned_model = fine_tuned_model.merge_and_unload()

# Reload tokenizer to save it
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, return_dict=True)
fine_tuned_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token
fine_tuned_tokenizer.padding_side = "right"

text = "Which Remote Services can I use for my vehicle in conjunction with the My BMW App?"

encoded_input = fine_tuned_tokenizer(text, return_tensors='pt')

save_model_path = Path('../models/vino_model.xml')

if not save_model_path.exists():
    ov_model = ov.convert_model(fine_tuned_model, example_input=dict(encoded_input))
    ov.save_model(ov_model, save_model_path)
else:
    raise Exception("model already exists!")

    
