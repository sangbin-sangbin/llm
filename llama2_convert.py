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
from converter import converters
from config import SUPPORTED_LLM_MODELS

model_configuration = SUPPORTED_LLM_MODELS['llama-2-chat-7b']
nncf.set_log_level(logging.ERROR)

pt_model_id = model_configuration["model_id"]
pt_model_name = 'llama-2-chat-7b'.split("-")[0]
model_type = AutoConfig.from_pretrained('../models/llama-2-7b-chat-hf', trust_remote_code=True).model_type
fp16_model_dir = Path('../models')

def convert_to_fp16():
    if (fp16_model_dir / "openvino_model.xml").exists():
        return
    if not model_configuration["remote"]:
        print(1111111)
        ov_model = OVModelForCausalLM.from_pretrained(
            '../models/llama-2-7b-chat-hf', export=True, compile=False
        )
        ov_model.half()
        ov_model.save_pretrained(fp16_model_dir)
        del ov_model
    else:
        print(2222222)
        model_kwargs = {}
        if "revision" in model_configuration:
            model_kwargs["revision"] = model_configuration["revision"]
        model = AutoModelForCausalLM.from_pretrained(
            '../models/llama-2-7b-chat-hf',
            torch_dtype=torch.float32,
            trust_remote_code=True,
            **model_kwargs
        )
        converters[pt_model_name](model, fp16_model_dir)
        del model
    gc.collect()

convert_to_fp16()
