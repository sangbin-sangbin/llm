import openvino as ov
import ipywidgets as widgets
from pathlib import Path


save_model_path = Path('./vino_model.xml')

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

compiled_model = core.compile_model(save_model_path, device.value)

# Compiled model call is performed using the same parameters as for the original model
res = compiled_model(encoded_input.data)[0]
print(res)