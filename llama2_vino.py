from ov_llm_model import model_classes


core = ov.Core()
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="CPU",
    description="Device:",
    disabled=False,
)

available_models = []
if int4_model_dir.exists():
    available_models.append("INT4")
if int8_model_dir.exists():
    available_models.append("INT8")
if fp16_model_dir.exists():
    available_models.append("FP16")

model_to_run = widgets.Dropdown(
    options=available_models,
    value=available_models[0],
    description="Model to run:",
    disabled=False,
)

from transformers import AutoTokenizer

if model_to_run.value == "INT4":
    model_dir = int4_model_dir
elif model_to_run.value == "INT8":
    model_dir = int8_model_dir
else:
    model_dir = fp16_model_dir
print(f"Loading model from {model_dir}")

model_name = model_configuration["model_id"]
class_key = model_id.value.split("-")[0]
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model_class = (
    OVModelForCausalLM
    if not model_configuration["remote"]
    else model_classes[class_key]
)
ov_model = model_class.from_pretrained(
    model_dir,
    device=device.value,
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
)

tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})
test_string = "2 + 2 ="
input_tokens = tok(test_string, return_tensors="pt", **tokenizer_kwargs)
answer = ov_model.generate(**input_tokens, max_new_tokens=2)
print(tok.batch_decode(answer, skip_special_tokens=True)[0])
