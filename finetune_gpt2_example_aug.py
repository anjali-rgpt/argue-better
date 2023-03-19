import os
import re
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
    def __init__(self, argument_list, example_list, topic_list, disctype_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        for argument, example, topic, disctype in zip(argument_list, example_list, topic_list, disctype_list):
            prep_argument = (f'<|startoftext|>Give a better example for this ineffective {disctype}'
                             f' : {argument}<|sep|>Better example : {example}<|endoftext|>')
            # tokenize 
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
    
def preprocess(df, cols):
    for col in cols: 
        # remove '\u00a0'|'\xa0':1198, '\n':2315
        df[col] = df[col].apply(lambda s: re.sub('\xA0', ' ', s))
        df[col] = df[col].apply(lambda s: s.replace('\n', ' '))
    
    print(sum(df['augmented_predictions'].str.contains('\u00a0')))
    print(sum(df['augmented_predictions'].str.contains('\xa0')))
    print(sum(df['augmented_predictions'].str.contains('\n')))

    # Keep the first 100 words from the content
    #news_df[exp_col] = news_df[exp_col].str.split(' ').apply(lambda x: ' '.join(x[:100]))

    return df
    
def load_dataset(tokenizer):
    # load dataset
    filepath = "data/effective/augmented_predictions_all.csv"
    df = pd.read_csv(filepath)
    df = preprocess(df, cols=['discourse_text', 'augmented_predictions'])
    df = df.sample(1000).reset_index()
    max_length = max([len(tokenizer.encode(text)) for text in df['discourse_text']])
    print("Max length: {}".format(max_length))
    
    # split 
    n = len(df)
    n_train = int(0.99 * n)
    indices = list(range(n))
    random.shuffle(indices)
    train_args = Subset(df['discourse_text'], indices[:n_train])
    val_args = Subset(df['discourse_text'], indices[n_train:])
    train_exps = Subset(df['augmented_predictions'], indices[:n_train])
    val_exps = Subset(df['augmented_predictions'], indices[n_train:])
    train_tpcs = Subset(df['topics'], indices[:n_train])
    val_tpcs = Subset(df['topics'], indices[n_train:])
    train_typs = Subset(df['discourse_type'], indices[:n_train])
    val_typs = Subset(df['discourse_type'], indices[n_train:])

     # generate class
    train_dataset = ExampleDataset(train_args, train_exps, train_tpcs, train_typs,
                                   tokenizer, max_length=tokenizer.model_max_length)
    
    return train_dataset, (val_args, val_exps, val_tpcs, val_typs)

torch.manual_seed(42)
model_name = "gpt2"
#model_name = "EleutherAI/gpt-neo-2.7B"
#special_tokens_dict = {'eos_token': <|endoftext|>, 
#                       'bos_token': <|startoftext|>, 
#                       'sep_token': <|sep|>, 
#                       'pad_token': <pad>}}
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

trainer.save_model("./models/gpt2/aug")

# eval

print("start evaluating")

# model = AutoModelForCausalLM.from_pretrained("./models/")

results = dict()
idx = 0

for argument, example, topic, disctype in tqdm(zip(val_dataset[0], val_dataset[1], val_dataset[2], val_dataset[3])):
    #prepare promp
    prep_argument = (f'<|startoftext|>Give a better example for this ineffective {disctype}'
                             f' : {argument}<|sep|>Better example : ')
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
                                    max_length=len(argument)*1.5,   #len
                                    top_p=0.95, 
                                    temperature=1.9, 
                                    num_return_sequences=20)

    pred = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    input = f'Give a better example for this ineffective {disctype} : {argument}Better example :'
    pred = pred.replace(input, '')
    results[idx] = {'input': argument, 
                    'pred': pred,
                    'true': example}
    idx += 1

    print(f'input: {argument}\npred: {pred}\ntrue: {example}')

#json_output = json.dumps(results, indent=4) 
with open("data/effective/finetune_gpt2_example_aug.json", "w") as outfile:
    json.dump(results, outfile)
#    outfile.write(json_output)
