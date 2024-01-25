from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
from pathlib import Path
import torch

#new_full_model = 'NousResearch/Llama-2-7b-chat-hf'
#new_full_model = '../models/llama-2-7b-chat-hf'
new_full_model = '../models/new-llama2-full-model'

vino_dir = '../models/llama2_vino'

ov_model = OVModelForCausalLM.from_pretrained(
    new_full_model, export=True, compile=False
)
ov_model.half()
ov_model.save_pretrained(vino_dir) 