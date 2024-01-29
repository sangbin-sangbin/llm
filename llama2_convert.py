from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
from pathlib import Path
import torch
from optimum.intel import OVQuantizer
import nncf


#new_full_model = 'NousResearch/Llama-2-7b-chat-hf'
#new_full_model = '../models/llama-2-7b-chat-hf'
new_full_model = '../models/new-llama2-full-model'

openvino_4_dir = '../models/llama2_openvino_4'
openvino_8_dir = '../models/llama2_openvino_8'
openvino_16_dir = '../models/llama2_openvino_16'

ov_model = OVModelForCausalLM.from_pretrained(
    new_full_model, export=True, compile=False
)
ov_model.half()

modelType = input("which type?: [int4 / int8 / fp16]")

if modelType == 'fp16':
    ov_model.save_pretrained(openvino_16_dir) 
elif modelType == 'int8':
    q = OVQuantizer.from_pretrained(ov_model)
    q.quantize(save_directory=openvino_8_dir, weights_only=True)
elif modelType == 'int4':
    model_compression_params = {
        "mode": nncf.CompressWeightsMode.INT4_SYM,
        "group_size": 128,
        "ratio": 0.8,
    }
    compressed_model = nncf.compress_weights(ov_model, **model_compression_params)
    ov.save_model(compressed_model, openvino_4_dir / "openvino_model.xml")
    shutil.copy(openvino_16_dir / "config.json", openvino_4_dir / "config.json")

