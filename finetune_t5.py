import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DEEPSPEED_ENABLE_PROFILING"] = "1"
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset, Subset, random_split
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, IntervalStrategy

def prepare_dataset(tokenizer):

    # load dataset
    dataset = load_dataset("a98zhang/ibm_argument_example")['train']
    
    prompt_template = f'Rewrite the following argument more effectively: {{input}}\nImproved argument: '
    prompt_length = len(tokenizer(prompt_template.format(input=""))["input_ids"])
    max_sample_length = tokenizer.model_max_length - prompt_length
    print(f"Prompt length: {prompt_length}")
    print(f"Max input length: {max_sample_length}")

    # The maximum total input sequence length after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = dataset.map(lambda x: tokenizer(x['argument'], truncation=True), batched=True, remove_columns=['argument', 'example', 'topic'])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    max_source_length = min(max_source_length, max_sample_length)
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = dataset.map(lambda x: tokenizer(x['example'], truncation=True), batched=True, remove_columns=['argument', 'example', 'topic'])
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # use 95th percentile as max target length
    max_target_length = int(np.percentile(target_lenghts, 95))
    print(f"Max target length: {max_target_length}")

    # split 
    dataset = dataset.train_test_split(test_size=0.01)

    # tokenize train dataset
    def preprocess(data):

        inputs = [prompt_template.format(input=arg) for arg in data['argument']]
        model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, padding='max_length', truncation=True)

        labels = tokenizer(text_target=data['example'], max_length=max_target_length, padding='max_length', truncation=True)
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

    # process dataset
    tokenized_trained = dataset["train"].map(preprocess, batched=True, remove_columns=['argument', 'example', 'topic'])
    
    return tokenized_trained, dataset["test"]

# load data, tokenizer, and model

torch.manual_seed(42)
model_name = "google/flan-t5-base"  #"gpt2"  "EleutherAI/gpt-neo-2.7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()

train_dataset, test_dataset = prepare_dataset(tokenizer)


# train

data_collator = DataCollatorForSeq2Seq(tokenizer)
         
generated = tokenizer("<|startoftext|>", return_tensors="pt").input_ids.cuda()

training_args = Seq2SeqTrainingArguments(output_dir='./results', 
                                  num_train_epochs=5, 
                                  logging_steps=500, 
                                  save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=2, 
                                  per_device_eval_batch_size=2, 
                                  warmup_steps=100,
                                  weight_decay=0.01, 
                                  logging_dir='./logs', 
                                  fp16=True, 
                                  deepspeed='./ds_config_flan_t5.json')
    
trainer = Seq2SeqTrainer(model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        data_collator = data_collator)

print("start training")

trainer.train()

trainer.save_model("./models")

# eval

print("start evaluating")

# model = AutoModelForCausalLM.from_pretrained("./models/")

for argument, example, topic in tqdm(zip(test_dataset[0], test_dataset[1], test_dataset[2])):
    #prepare promp
    prep_argument = f'<|startoftext|>Argument: {argument}\nArgue this idea on {topic} better:<|sep|>'
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

    print("Input: {}\n\nPred: {}\n\nTrue: {}\n\n\n\n\n".format(argument, pred, example))

                                           
