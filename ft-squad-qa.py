from datasets import load_dataset
from transformers import default_data_collator
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import os
os.environ['PJRT_DEVICE'] = 'GPU'
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

squad = load_dataset("squad_v2")
complete_data = concatenate_datasets([squad['train'], squad['validation']])

def filter_no_answers(example):
    return bool(example['answers']['text'])
def filter_no_question(example):
    return bool(example['question'])
# Filter out examples with no answers
filtered_dataset = complete_data.filter(filter_no_answers)
filtered_dataset = filtered_dataset.filter(filter_no_question)
complete_data = filtered_dataset
dataset_train_test = complete_data.class_encode_column("title").train_test_split(test_size=0.2, stratify_by_column="title", seed = 1)
squad = dataset_train_test


tokenizer = AutoTokenizer.from_pretrained("Dhairya/nlp-bert-wordWeavers-tokenizer")
model = AutoModelForQuestionAnswering.from_pretrained("Dhairya/nlp-bert-wordWeavers-pretrained").to(device)


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
data_collator = default_data_collator

train_dataset = tokenized_squad["train"]
test_dataset = tokenized_squad["test"]

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

print(trainer.evaluate())

model.save_pretrained("bert-wordweavers-ft-squad")
tokenizer.save_pretrained("bert-wordweavers-ft-squad-tokenizer")
model.push_to_hub("bert-wordweavers-ft-squad")
tokenizer.push_to_hub("bert-wordweavers-ft-squad-tokenizer")