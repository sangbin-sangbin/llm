from transformers import AutoConfig
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov
from pathlib import Path


model_type = AutoConfig.from_pretrained('../models/llama-2-7b-chat-hf', trust_remote_code=True).model_type
model_dir = Path('../models/llama2_vino')

ov_model = OVModelForCausalLM.from_pretrained(
    '../models/llama-2-7b-chat-hf', export=True, compile=False
)
ov_model.half()
ov_model.save_pretrained(model_dir) 