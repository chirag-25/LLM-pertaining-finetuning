from sklearn.metrics import accuracy_score
import torch
from transformers import TrainingArguments, Trainer
import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer


os.environ['PJRT_DEVICE'] = 'GPU'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


sst_model = AutoModelForSequenceClassification.from_pretrained("Dhairya/nlp-bert-wordWeavers-pretrained", num_labels=2).to(device)
sst_tokenizer = AutoTokenizer.from_pretrained("Dhairya/nlp-bert-wordWeavers-tokenizer")
sst2_dataset = load_dataset('glue', 'sst2')
sst2_dataset_train = sst2_dataset['train']
dataset = sst2_dataset_train.train_test_split(test_size=0.2, stratify_by_column="label" , seed=1)

train_dataset = dataset['train'].map(lambda examples: {'labels': examples['label']}, batched=True)
test_dataset = dataset['test'].map(lambda examples: {'labels': examples['label']}, batched=True)

test_dataset = test_dataset.remove_columns(['label'])
train_dataset = train_dataset.remove_columns(['label'])

MAX_LENGTH = 128
train_dataset = train_dataset.map(lambda e: sst_tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
test_dataset = test_dataset.map(lambda e: sst_tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return sst_tokenizer(examples['sentence'], padding="max_length", truncation=True)


#tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='model_preds',
    learning_rate=1e-3,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    do_train=True,
    do_eval=False,
    load_best_model_at_end=False,
    # eval_steps=100,
    save_strategy = "epoch",
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall}


trainer = Trainer(
    model = sst_model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate(eval_dataset = test_dataset))

sst_model.save_pretrained("nlp-bert-wordweavers-ft-sst2")
sst_tokenizer.save_pretrained("nlp-bert-wordweavers-ft-sst2-tokenizer")
sst_model.push_to_hub("nlp-bert-wordweavers-ft-sst2")
sst_tokenizer.push_to_hub("nlp-bert-wordweavers-ft-sst2-tokenizer")