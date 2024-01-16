import openvino as ov
import ipywidgets as widgets
from pathlib import Path
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


model_name = "NousResearch/Llama-2-7b-chat-hf"

save_model_path = Path('../Desktop/vino_model.xml')

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

compiled_model = core.compile_model(save_model_path, device.value)


ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

ov_model = OVModelForCausalLM.from_pretrained(
    model_name,
    device=device.value,
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_name, trust_remote_code=True),
    trust_remote_code=True,
)





fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, return_dict=True)
fine_tuned_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
fine_tuned_tokenizer.pad_token = fine_tuned_tokenizer.eos_token
fine_tuned_tokenizer.padding_side = "right"

while True:
    text = "How can I buy BMW?"#input("question: ")
    encoded_input = fine_tuned_tokenizer(text, return_tensors='pt')
    res = compiled_model.generate(**encoded_input, max_new_tokens=2)
    print(fine_tuned_tokenizer.batch_decode(res, skip_special_tokens=True)[0])
