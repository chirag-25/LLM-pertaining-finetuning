from datasets import load_dataset
from transformers import AutoTokenizer
import random
from transformers import TrainerCallback, TrainerState, TrainerControl
import math
from transformers import BertConfig, BertForPreTraining, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch
import os
os.environ['PJRT_DEVICE'] = 'GPU'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
datasets_cleaned = datasets.filter(lambda example: len(example['text'])  > 0 and not example["text"].startswith(" ="))

# create a python generator to dynamically load the data
def batch_iterator(batch_size=10000):
    for i in range(0, len(datasets_cleaned['train']), batch_size):
        yield datasets_cleaned['train'][i : i + batch_size]["text"]


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=25000)
# tokenizer.save_pretrained("tokenizer")

def tokenization(example):
    return tokenizer(example["text"], add_special_tokens=False)

datasets_tokenized = datasets_cleaned.map(
    tokenization, batched=True, num_proc=4)

datasets_tokenized = datasets_tokenized.map(batched= True, remove_columns='text')

def concatenation(example):
    BLOCK_SIZE= 255
    # concatenating the inputs_ids, token_ids, attention_mask respectively to a list to create a single list of all the tokens
    concatenated_examples = {}
    for keys in example.keys(): # inputs_ids, token_ids, attention_mask
        concatenated_examples[keys] = sum(example[keys], [])

    # total length same across all the keys
    n = len(concatenated_examples[list(example.keys())[0]])
    n = (n//BLOCK_SIZE) * BLOCK_SIZE

    # breaking the total combined list to get BLOCK SIZE chunks
    result = {}
    for keys, token_type in concatenated_examples.items(): 
        result[keys] = []
        for i in range(0, n, BLOCK_SIZE):
            result[keys].append(token_type[i: i+BLOCK_SIZE])
    return result

datasets_block_size = datasets_tokenized.map(
    concatenation, batched=True, batch_size=1000, num_proc=4)

def preparing_NSP_dataset(example, ind, dataset, n):
    # NSP dataset is created such that 50% sentence 2 comes after sentence 1. Rest 50% it is random
    sent_1 = example['input_ids']
    attention_mask = [1] * 512
    next_sentence_label = 1

    if ind % 2 == 0: 
        next_ind = ind + 1
        if next_ind < len(dataset['input_ids']): 
            sent_2 = dataset['input_ids'][next_ind]
        else: # last sentence has no next sentence
            next_ind = random.randint(0, n-1)
            sent_2 = dataset['input_ids'][next_ind]
            next_sentence_label = 0

    else:
        next_sentence_label = 0
        next_ind = random.randint(0, n-1) # randomly choosing the next index
        if next_ind == ind + 1: # if randomly choosed the next sentence then changing the next sentence label
            next_sentence_label = 1
        sent_2 = dataset['input_ids'][next_ind]
        
    # input  =  [cls] + sent1 + [sep] + sent2
    input_ids = [tokenizer.cls_token_id] + sent_1 + [tokenizer.sep_token_id] + sent_2
    token_type_ids = [0] * (257) + [1] * (255)
    attention_mask = [1] + example['attention_mask'] + [1] + dataset[next_ind]['attention_mask']
    
    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'next_sentence_label': next_sentence_label
    }

dataset_train = datasets_block_size['train']
dataset_validation = datasets_block_size['validation']
dataset_test = datasets_block_size['test']

dataset_NSP_train = dataset_train.map(
    lambda example, ind: preparing_NSP_dataset(example, ind, dataset_train, n =len(dataset_train)),with_indices=True, num_proc=32)

dataset_NSP_validation = dataset_validation.map(
    lambda example, ind: preparing_NSP_dataset(example, ind, dataset_validation, n =len(dataset_validation)),with_indices=True, num_proc=32)

dataset_NSP_test = dataset_test.map(
    lambda example, ind: preparing_NSP_dataset(example, ind, dataset_test, n =len(dataset_test)),with_indices=True, num_proc=32)

class PerplexityCallback(TrainerCallback):
    def __init__(self, model):
        self.epoch = 0
        self.model = model
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.epoch += 1
        print(f"Epoch {self.epoch}")
        print(f"state.log_history: {state.log_history}")


# Assuming you have a test dataset named dataset_NSP_test
collater_pt = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors="pt"
)

# Initialize the model
config = BertConfig()
model = BertForPreTraining(config).to(device)

def randomize_weights(model):
    for param in model.parameters():
        if param.requires_grad:
            param.data = torch.randn(param.size()).to(device)

randomize_weights(model)

perplexity_callback = PerplexityCallback(model)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="nlp-bert-wordweavers-v5",
    num_train_epochs=5,
    learning_rate=0.1,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    logging_first_step=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_NSP_train,
    eval_dataset=dataset_NSP_test,
    data_collator=collater_pt,
    tokenizer=tokenizer,
    callbacks=[perplexity_callback]
)

trainer.train()

model.save_pretrained("nlp-bert-wordWeavers-pretrained")
tokenizer.save_pretrained("nlp-bert-wordWeavers-tokenizer")
model.push_to_hub("nlp-bert-wordWeavers-pretrained")
tokenizer.push_to_hub("nlp-bert-wordWeavers-tokenizer")


