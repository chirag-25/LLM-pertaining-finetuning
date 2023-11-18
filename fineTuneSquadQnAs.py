import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
import random
import os
os.environ['PJRT_DEVICE'] = 'GPU'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

t1 = AutoTokenizer.from_pretrained("Dhairya/nlp-bert-wordWeavers-tokenizer")
squad_model = AutoModelForQuestionAnswering.from_pretrained("Dhairya/nlp-bert-wordWeavers-pretrained").to(device)

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    if context != '' and question != '' and answer['text'] != '':
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

    return contexts, questions, answers

train_contexts, train_questions, train_answers = read_squad('squad/squad/train-v2.0.json')
val_contexts, val_questions, val_answers = read_squad('squad/squad/dev-v2.0.json')

tot_size = len(train_contexts)
all_indices = list(range(tot_size))
percentage_to_sample = 20
num_indices_to_sample = int(len(all_indices) * (percentage_to_sample / 100.0))

random_sampled_indices = random.sample(all_indices, num_indices_to_sample)

test_contexts = [train_contexts[i] for i in random_sampled_indices]
test_questions = [train_questions[i] for i in random_sampled_indices]
test_answers = [train_answers[i] for i in random_sampled_indices]

train_contexts = [train_contexts[i] for i in range(tot_size) if i not in random_sampled_indices]
train_questions = [train_questions[i] for i in range(tot_size) if i not in random_sampled_indices]
train_answers = [train_answers[i] for i in range(tot_size) if i not in random_sampled_indices]

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        # fixing the dataset when it is off my 1/2 characters
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2

add_end_idx(train_answers, train_contexts)
add_end_idx(test_answers, test_contexts)

train_encodings = t1(train_contexts, train_questions, truncation=True, padding=True)
test_encodings = t1(test_contexts, test_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        if start_positions[-1] is None:
            start_positions[-1] = t1.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = t1.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(test_encodings, test_answers)

# Defining our dataset using pytorch
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
    
train_dataset = SquadDataset(train_encodings)
test_dataset = SquadDataset(test_encodings)


squad_model.train()
no_of_epochs = 3
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#using AdamW as the optimizer in order to avoid overfitting
optim = AdamW(squad_model.parameters(), lr=5e-3,no_deprecation_warning=True)

for epoch in range(no_of_epochs):
    print("Epoch {}".format(epoch))
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = squad_model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
squad_model.eval()


squad_model.save_pretrained("nlp-bert-wordweavers-ft-squad")
t1.save_pretrained("nlp-bert-wordweavers-ft-squad-tokenizer")
squad_model.push_to_hub("nlp-bert-wordweavers-ft-squad")
t1.push_to_hub("nlp-bert-wordweavers-ft-squad-tokenizer")