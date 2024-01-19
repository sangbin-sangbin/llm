from transformers import AutoTokenizer, AutoConfig
import openvino as ov
from pathlib import Path
from optimum.intel.openvino import OVModelForCausalLM
import time


core = ov.Core()
device = "CPU"
model_dir = Path('../models/llama2_vino')
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

tokenizer = AutoTokenizer.from_pretrained('../models/llama-2-7b-chat-hf', trust_remote_code=True)

ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device=device,
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
)

test_strings = [
    "Which Remote Services can I use for my vehicle in conjunction with the My BMW App?",
    "Who are you?"
]
for test_string in test_strings:
    input_tokens = tokenizer(f"<s>[INST] {test_string} [/INST] ", return_tensors="pt", add_special_tokens=False)
    start = time.time()
    result = ov_model.generate(**input_tokens, max_new_tokens=1024)
    end = time.time()
    print(tokenizer.batch_decode(result, skip_special_tokens=True)[0])
    print("elapsed time:", end - start)

while True:
    question = input("question: ")
    input_tokens = tokenizer(f"<s>[INST] {question} [/INST] ", return_tensors="pt", add_special_tokens=False)
    start = time.time()
    answer = ov_model.generate(**input_tokens, max_new_tokens=1024)
    end = time.time()
    print(tokenizer.batch_decode(answer, skip_special_tokens=True)[0])
    print("elapsed time:", end - start)
