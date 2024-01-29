from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardTrainer
from datasets import Dataset


model_name = "distilroberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def formatting_func(examples):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
    prompt_plus_chosen_response = examples["instruction"] + "\n" + examples["chosen_response"]
    prompt_plus_rejected_response = examples["instruction"] + "\n" + examples["rejected_response"]
    tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }

prepared_dataset = {'instruction':['hello', 'hello', 'hello'], 'chosen_response':['hi', 'hi', 'hi'], 'rejected_response':['I hate you', 'I hate you', 'I hate you']}
prepared_dataset = Dataset.from_dict(prepared_dataset)

formatted_dataset = prepared_dataset.map(formatting_func)
formatted_dataset = formatted_dataset.train_test_split()
# Configuring the training arguments
training_args = TrainingArguments(
    output_dir="../../models/reward_model",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    logging_steps=1,
    num_train_epochs = 10,
    report_to=None,
)
# Loading the RewardTrainer from TRL
trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
)
trainer.train()
trainer.save_model()

model = AutoModelForSequenceClassification.from_pretrained("../../models/reward_model", num_labels=1)

pt = tokenizer(
    "hello\nhi",
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

print(model(**pt)['logits'])