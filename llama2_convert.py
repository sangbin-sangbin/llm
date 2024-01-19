from transformers import AutoModelForCausalLM, AutoConfig
from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov
from pathlib import Path
import shutil
import torch
import logging
import nncf
import gc
from config import SUPPORTED_LLM_MODELS


model_configuration = SUPPORTED_LLM_MODELS['llama-2-chat-7b']
nncf.set_log_level(logging.ERROR)

pt_model_name = 'llama-2-chat-7b'.split("-")[0]
model_type = AutoConfig.from_pretrained('../models/llama-2-7b-chat-hf', trust_remote_code=True).model_type
fp16_model_dir = Path('../models/llama2_vino')

ov_model = OVModelForCausalLM.from_pretrained(
    '../models/llama-2-7b-chat-hf', export=True, compile=False
)
ov_model.half()
ov_model.save_pretrained(fp16_model_dir)