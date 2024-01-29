from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
from tqdm import tqdm
from transformers import pipeline

batch_size = 4

config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=batch_size
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}


reward_model = AutoModelForSequenceClassification.from_pretrained("../../models/reward_model", num_labels=1).to('cuda')

def tokenize(data):
    tmp = tokenizer(
        data,
        padding=True,
        truncation=True,
        max_length=1024,
    )['input_ids']
    return list(map(lambda x : torch.tensor(x).to('cuda'), tmp))

datalist = ['hello my name is sangbin', 'what is your name?', 'hello my name is sangbin', 'what is your name?', 'hello my name is sangbin', 'what is your name?', 'hello my name is sangbin', 'what is your name?']

dataset = [datalist[i:i+batch_size] for i in range(0, len(datalist), batch_size)]
dataset = list(map(tokenize, dataset))

enter = torch.tensor(tokenizer('\n')['input_ids']).to('cuda')

for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
    for batch in dataset: 

        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(batch, **generation_kwargs)
        #responses = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        #### Compute reward score
        texts = [torch.cat((q, enter, r), dim=0) for q, r in zip(batch, response_tensors)]

        attention_mask = []
        rewards = []
        for text in texts:
            pt = {'input_ids': text.unsqueeze(0).to('cuda'), 'attention_mask': torch.tensor([[1]*len(text)]).to('cuda')}
            rewards.append(reward_model(**pt)['logits'][0])

        #### Run PPO step
        stats = ppo_trainer.step(batch, response_tensors, rewards)
        
#### Save model
ppo_trainer.save_pretrained("../../models/my_ppo_model")