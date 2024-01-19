from transformers import AutoTokenizer, AutoConfig
import openvino as ov
from pathlib import Path
from optimum.intel.openvino import OVModelForCausalLM
from config import SUPPORTED_LLM_MODELS


core = ov.Core()
device = "CPU"
model_dir = Path('../models/llama2_vino')

ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

tok = AutoTokenizer.from_pretrained('../models/llama-2-7b-chat-hf', trust_remote_code=True)

ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device=device,
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
)

while True:
    test_string = input("question: ")
    input_tokens = tok(f"<s>[INST] {test_string} [/INST] ", return_tensors="pt", add_special_tokens=False)
    answer = ov_model.generate(**input_tokens, max_new_tokens=1024)
    print(tok.batch_decode(answer, skip_special_tokens=True)[0])
