from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
from pathlib import Path
import torch
from peft import PeftModel


model_name = "../models/llama-2-7b-chat-hf"

new_model = "../models/new-llama2-model"

vino_dir = Path('../models/llama2_vino')

device_map = {"": 0}

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
ov_model = OVModelForCausalLM.from_pretrained(
    model_name, export=True, compile=False
)

fine_tuned_model = PeftModel.from_pretrained(ov_model, new_model)
fine_tuned_model = fine_tuned_model.merge_and_unload()

# Reload tokenizer to save it
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
fine_tuned_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token
fine_tuned_tokenizer.padding_side = "right"



ov_model = OVModelForCausalLM.from_pretrained(
    fine_tuned_model, export=True, compile=False
)
ov_model.half()
ov_model.save_pretrained(vino_dir) 