import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DEEPSPEED_ENABLE_PROFILING"] = "1"
import pandas as pd
from tqdm import tqdm
import random
import json

import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

# Dataset Class
class ExampleDataset(Dataset):
    def __init__(self, argument_list, example_list, topic_list, disctype_list, context_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        for argument, example, topic, disctype, context in zip(argument_list, example_list, topic_list, disctype_list, context_list):
            prep_argument = f'<|startoftext|>Topic name: {topic} <|sep|> Discourse type: {disctype} <|sep|> Context: {context} <|sep|> Argument: {argument} \n<|sep|>Better example: {example}<|endoftext|>'
            encodings_dict = tokenizer(prep_argument, 
                                       truncation=True,
                                       max_length = max_length, 
                                       padding="max_length")
            # append to list
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

            # when training, the input data will be passed in also as the label 
            # because we are training a language model and we want the model to
            # learn the pattern of argument + example struture
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
    
def load_dataset(tokenizer):
    # load dataset
    filepath = "augmented_predictions_all.csv"
    df = pd.read_csv(filepath)
    df = df.sample(1000).reset_index()
    max_length = max([len(tokenizer.encode(text)) for text in df['discourse_text']]) + max([len(tokenizer.encode(text)) for text in df['context_predictions']]) + 150
    print("Max length: {}".format(max_length))
    
    # split 
    n = len(df)
    n_train = int(0.99 * n)
    indices = list(range(n))
    random.shuffle(indices)
    train_args = Subset(df['discourse_text'], indices[:n_train])
    val_args = Subset(df['discourse_text'], indices[n_train:])
    train_exps = Subset(df['predictions'], indices[:n_train])
    val_exps = Subset(df['predictions'], indices[n_train:])
    train_tpcs = Subset(df['topics'], indices[:n_train])
    val_tpcs = Subset(df['topics'], indices[n_train:])
    train_typs = Subset(df['discourse_type'], indices[:n_train])
    val_typs = Subset(df['discourse_type'], indices[n_train:])
    train_cnts = Subset(df['context_predictions'], indices[:n_train])
    val_cnts = Subset(df['context_predictions'], indices[n_train:])


     # generate class
    train_dataset = ExampleDataset(train_args, train_exps, train_tpcs, train_typs, train_cnts
                                   tokenizer, max_length=tokenizer.model_max_length)
    
    return train_dataset, (val_args, val_exps, val_tpcs, val_typs, val_cnts)

torch.manual_seed(42)
model_name = "gpt2"
#model_name = "EleutherAI/gpt-neo-2.7B"
#special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}
#tokenizer_orig.add_special_tokens(special_tokens_dict)
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          bos_token='<|startoftext|>', 
                                          eos_token='<|endoftext|>', 
                                          sep_token='<|sep|>',
                                          pad_token='<pad>')

model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
model.resize_token_embeddings(len(tokenizer))

train_dataset, val_dataset = load_dataset(tokenizer)

# train

training_args = TrainingArguments(output_dir='./results', 
                                  num_train_epochs=5, 
                                  logging_steps=500, 
                                  save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=2, 
                                  per_device_eval_batch_size=2, 
                                  warmup_steps=100,
                                  weight_decay=0.01, 
                                  logging_dir='./logs', 
                                  fp16=True, 
                                  deepspeed='./ds_config.json')
    
trainer = Trainer(model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[0] for f in data])})

print("start training")

trainer.train()

trainer.save_model("./models/gpt2/context")

# eval

print("start evaluating")

# model = AutoModelForCausalLM.from_pretrained("./models/")

results = dict()
idx = 0

for argument, example, topic, disctype, context in tqdm(zip(val_dataset[0], val_dataset[1], val_dataset[2], val_dataset[3], val_dataset[4])):
    #prepare promp
    prep_argument = f'<|startoftext|>Topic name: {topic} <|sep|> Discourse type: {disctype} <|sep|> Context: {context} <|sep|> Argument: {argument} \n<|sep|>Better example: '
    generated = tokenizer(prep_argument, 
                      return_tensors="pt").input_ids.cuda()
    #generate
    sample_outputs = model.generate(generated, 
                                    do_sample=True, 
                                    top_k=50,
                                    bos_token='<|startoftext|>',
                                    eos_token='<|endoftext|>',
                                    sep_token='<|sep|>',
                                    pad_token='<pad>',
                                    max_length=len(argument), 
                                    top_p=0.95, 
                                    temperature=1.9, 
                                    num_return_sequences=20)

    pred = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    results[idx] = {'input': argument, 
                    'pred': pred,
                    'true': example}
    idx += 1

json_output = json.dumps(results, indent=4) 
with open("data/effective/finetune_gpt2_example_context.json", "w") as outfile:
        outfile.write(json_output)
