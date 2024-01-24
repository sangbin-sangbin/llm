import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    EarlyStoppingCallback
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from random import shuffle
import json
from datasets import Dataset


# The model that you want to train from the Hugging Face hub
model_name = "microsoft/phi-2"


################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1000

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 100

# Log every X updates steps
logging_steps = 100

eval_steps = 100

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

# Load dataset (you can process it here)
aug_type = input('which data type? [no / bert / llm]\n>>> ')
if aug_type == 'no':
    train_data_list = json.load(open('../data/no_augmented_data.json'))
    new_model = "../models/new-phi2-model-no-aug"
elif aug_type == 'bert':
    train_data_list = json.load(open('../data/bert_augmented_data.json'))
    new_model = "../models/new-phi2-model-bert-aug"
else:
    train_data_list = json.load(open('../data/llm_augmented_data.json'))
    new_model = "../models/new-phi2-model-llama-aug"

shuffle(train_data_list)
train_data_dict = {"text": [item["text"] for item in train_data_list]}
train_dataset = Dataset.from_dict(train_data_dict)

seen_test_data_list = json.load(open('../data/seen_test_data.json'))
seen_test_data_len = len(seen_test_data_list)
val_ratio = 0.3
val_data_list = seen_test_data_list[:int(seen_test_data_len*val_ratio)]
val_dataset = Dataset.from_dict({"text": [item["text"] for item in val_data_list]})

seen_data_list = seen_test_data_list[int(seen_test_data_len*val_ratio):]
seen_test_dataset = Dataset.from_dict({"text": [item["text"] for item in seen_data_list]})

unseen_test_data_list = json.load(open('../data/unseen_test_data.json'))
unseen_test_dataset = Dataset.from_dict({"text": [item["text"] for item in unseen_test_data_list]})

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    bias="none",
    lora_dropout=0.05, # Conventional
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    do_eval=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
)

def tokenize(element):
    outputs = tokenizer(
        element['text'],
        add_special_tokens=True,
        truncation=True,
        padding=False,
        max_length=1024,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

tokenized_unseen_test_dataset = unseen_test_dataset.map(
    tokenize,
    batched=True,
    remove_columns=unseen_test_dataset.column_names
)
res = trainer.evaluate(eval_dataset=tokenized_unseen_test_dataset, metric_key_prefix='unseen_test')
print('loss_for_unseen_data:', res['unseen_test_loss'])

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

tokenized_seen_test_dataset = seen_test_dataset.map(
    tokenize,
    batched=True,
    remove_columns=seen_test_dataset.column_names
)
res = trainer.evaluate(eval_dataset=tokenized_seen_test_dataset, metric_key_prefix='seen_test')
print('loss_for_seen_data:', res['seen_test_loss'])

tokenized_unseen_test_dataset = unseen_test_dataset.map(
    tokenize,
    batched=True,
    remove_columns=unseen_test_dataset.column_names
)
res = trainer.evaluate(eval_dataset=tokenized_unseen_test_dataset, metric_key_prefix='unseen_test')
print('loss_for_unseen_data:', res['unseen_test_loss'])
