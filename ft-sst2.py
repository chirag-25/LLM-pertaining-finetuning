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



# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return sst_tokenizer(examples['sentence'], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='model_preds',
    learning_rate=1e-3,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    do_train=True,
    do_eval=False,
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

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
test_dataset = tokenized_datasets["test"].shuffle(seed=42)

trainer = Trainer(
    model = sst_model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(eval_dataset=test_dataset)

sst_model.save_pretrained("bert-wordweavers-ft-sst2")
sst_tokenizer.save_pretrained("bert-wordweavers-ft-sst2-tokenizer")
sst_model.push_to_hub("bert-wordweavers-ft-sst2")
sst_tokenizer.push_to_hub("bert-wordweavers-ft-sst2-tokenizer")