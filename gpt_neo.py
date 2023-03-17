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

import torch
from torch.utils.data import Dataset, random_split
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

# Dataset Class
class ExampleDataset(Dataset):
    def __init__(self, argument_list, example_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        for argument, example in zip(argument_list, example_list):
            prep_argument = f'<startoftext>Argument: {argument}\nBetter example: {example}<endoftext>'
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
    
def load_dataset(tokenizer):
    # load dataset
    filepath = "data/IBM_Debater_(R)_arg_quality_rank_30k/arg_quality_rank_30k_examples.csv"
    df = pd.read_csv(filepath)
    arguments = df['argument']
    examples = df['example']
    max_length = max([len(tokenizer.encode(txt)) for txt in (arguments + examples)])
    print("Max length: {}".format(max_length))

    # split 
    n = len(arguments)
    n_train = int(0.99 * n)
    train_args, val_args = random_split(arguments, [n_train, n-n_train])
    train_exps, val_exps = random_split(examples, [n_train, n-n_train])

     # generate class
    train_dataset = ExampleDataset(train_args, train_exps, 
                                   tokenizer, max_length=max_length)
    
    return train_dataset, (val_args, val_exps)

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", 
                                          bos_token='<startoftext>', 
                                          eos_token='<endoftext>', 
                                          pad_token='<pad>')

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").cuda()
model.resize_token_embeddings(len(tokenizer))

train_dataset, val_dataset = load_dataset(tokenizer)

# train

training_args = TrainingArguments(output_dir='./results', 
                                  num_train_epochs=4.3, 
                                  logging_steps=500, 
                                  save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=15, 
                                  per_device_eval_batch_size=15, 
                                  warmup_steps=50,
                                  weight_decay=0.01, 
                                  logging_dir='./logs', 
                                  fp16=True, 
                                  deepspeed='./ds_config_gpt_neo_27.json')
    
trainer = Trainer(model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[0] for f in data])})

print("start training")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        trainer.train()

prof.export_chrome_trace("profile.json")

# eval

for argument, example in tqdm(zip(val_dataset[0], val_dataset[1])):
    #prepare promp
    prep_argument = f'<startoftext>Argument: {argument}\nBetter example:'
    generated = tokenizer("<startoftext>", 
                      return_tensors="pt").input_ids.cuda()
    #generate
    sample_outputs = model.generate(generated, 
                                    do_sample=True, 
                                    top_k=50,
                                    bos_token='<startoftext>',
                                    eos_token='<endoftext>', 
                                    pad_token='<pad>',
                                    max_length=300, 
                                    top_p=0.95, 
                                    temperature=1.9, 
                                    num_return_sequences=20)

    pred = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

    print("Input: {}\nPred: {}\nTrue: {}\n\n".format(argument, pred, example))

                                           
