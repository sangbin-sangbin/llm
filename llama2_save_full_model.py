from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
from pathlib import Path
import torch
from peft import PeftModel


model_name = "../models/llama-2-7b-chat-hf"

new_model = "../models/new-llama2-model-llama-aug"

new_full_model = "../models/new-llama2-full-model"

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

fine_tuned_model.save_pretrained(new_full_model)