from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
from pathlib import Path
import torch


new_full_model = "../models/new-llama2-full-model"

vino_dir = Path('../models/llama2_vino')

device_map = {"": 0}

# Reload model in FP16 and merge it with LoRA weights
fine_tuned_model = AutoModelForCausalLM.from_pretrained(
    new_full_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

ov_model = OVModelForCausalLM.from_pretrained(
    fine_tuned_model, export=True, compile=False
)
ov_model.half()
ov_model.save_pretrained(vino_dir) 